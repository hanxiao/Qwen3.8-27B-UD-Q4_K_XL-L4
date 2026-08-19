// Capture MTP draft-head training data from a served Qwen3.8-27B GGUF.
//
// The nextn (MTP) head consumes h_nextn, and in src/models/qwen35.cpp h_nextn is the tensor
// tagged "result_norm" one line later -- i.e. the ordinary post-output_norm hidden state. So
// capturing training data needs no patch: ask for logits at every position and read the standard
// embeddings and logits APIs. This runs at prefill speed rather than decode speed, which is the
// difference between hours and days.
//
// Per position we record what the head must learn to reproduce:
//   tok      the input token at this position                       int32
//   h        the hidden state that will be fed to the head          [n_embd] bf16
//   argmax   the target's own committed token, which IS the accept  int32
//            criterion at inference time, so it is the label
//   topk     top-K (id, logit) for distribution distillation        K*(int32,float)
//
// The input token is recorded because the head is not a function of h alone: it consumes the
// hidden state at i together with the embedding of the token at i+1, and predicts the token at
// i+2. Reconstructing that sequence by re-tokenizing the corpus afterwards has to reproduce the
// chunking and the BOS handling exactly, so the ids are written here instead.
//
// Usage: capture-mtp-data -m model.gguf -f corpus.txt -o out.bin [-k 8] [-c 8192] [-n MAX_TOK]
#include "llama.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

static uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    // round-to-nearest-even on the discarded low half
    const uint32_t rounding = 0x7fff + ((u >> 16) & 1);
    return (uint16_t) ((u + rounding) >> 16);
}

int main(int argc, char ** argv) {
    std::string model_path, corpus_path, out_path;
    int topk = 8, n_ctx = 8192;
    long max_tokens = -1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : nullptr; };
        if      (a == "-m") model_path  = next() ?: "";
        else if (a == "-f") corpus_path = next() ?: "";
        else if (a == "-o") out_path    = next() ?: "";
        else if (a == "-k") topk        = atoi(next() ?: "8");
        else if (a == "-c") n_ctx       = atoi(next() ?: "8192");
        else if (a == "-n") max_tokens  = atol(next() ?: "-1");
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || corpus_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f corpus.txt -o out.bin [-k 8] [-c 8192] [-n MAX]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n_ctx;
    cparams.n_batch  = n_ctx;
    cparams.n_ubatch = 512;
    cparams.embeddings = true;               // we want per-token hidden states
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // read and tokenize the corpus
    FILE * cf = fopen(corpus_path.c_str(), "rb");
    if (!cf) { fprintf(stderr, "cannot open %s\n", corpus_path.c_str()); return 1; }
    std::string text;
    { char buf[1 << 16]; size_t r; while ((r = fread(buf, 1, sizeof buf, cf)) > 0) text.append(buf, r); }
    fclose(cf);

    std::vector<llama_token> toks(text.size() + 8);
    int n_toks = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (n_toks < 0) { toks.resize(-n_toks); n_toks = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false); }
    toks.resize(n_toks);
    if (max_tokens > 0 && (long) toks.size() > max_tokens) toks.resize(max_tokens);
    fprintf(stderr, "corpus: %zu tokens, n_embd=%d n_vocab=%d topk=%d\n", toks.size(), n_embd, n_vocab, topk);

    FILE * of = fopen(out_path.c_str(), "wb");
    if (!of) { fprintf(stderr, "cannot open %s for writing\n", out_path.c_str()); return 1; }
    // header: magic, n_embd, topk, n_positions (patched at the end)
    // MTPCAP2 adds the input token id ahead of the hidden state; a reader must not accept
    // a v1 file as if it were this layout.
    const char magic[8] = "MTPCAP2";
    int32_t hdr[3] = { n_embd, topk, 0 };
    fwrite(magic, 1, 8, of); fwrite(hdr, sizeof(int32_t), 3, of);

    std::vector<uint16_t> hbuf(n_embd);
    std::vector<int32_t>  kid(topk);
    std::vector<float>    kval(topk);
    std::vector<int>      idx(n_vocab);

    const int chunk = n_ctx - 8;
    long written = 0;
    for (size_t off = 0; off + 2 < toks.size(); off += chunk) {
        const int n = (int) std::min((size_t) chunk, toks.size() - off);
        if (n < 2) break;

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]     = toks[off + i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;          // every position, not just the last
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %zu\n", off); llama_batch_free(batch); break; }

        for (int i = 0; i < n; i++) {
            const float * emb = llama_get_embeddings_ith(ctx, i);
            const float * lg  = llama_get_logits_ith(ctx, i);
            if (!emb || !lg) continue;

            for (int j = 0; j < n_embd; j++) hbuf[j] = f32_to_bf16(emb[j]);

            for (int v = 0; v < n_vocab; v++) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&](int a, int b) { return lg[a] > lg[b]; });
            for (int t = 0; t < topk; t++) { kid[t] = idx[t]; kval[t] = lg[idx[t]]; }

            const int32_t argmax = kid[0];   // the accept criterion, and so the label
            const int32_t tok_in = toks[off + i];
            fwrite(&tok_in,     sizeof(int32_t), 1, of);
            fwrite(hbuf.data(), sizeof(uint16_t), n_embd, of);
            fwrite(&argmax,     sizeof(int32_t), 1, of);
            fwrite(kid.data(),  sizeof(int32_t), topk, of);
            fwrite(kval.data(), sizeof(float),   topk, of);
            written++;
        }
        llama_batch_free(batch);
        fprintf(stderr, "\r%ld positions", written); fflush(stderr);
    }
    fprintf(stderr, "\n");

    hdr[2] = (int32_t) written;
    fseek(of, 8, SEEK_SET); fwrite(hdr, sizeof(int32_t), 3, of);
    fclose(of);
    fprintf(stderr, "wrote %ld positions to %s\n", written, out_path.c_str());

    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
