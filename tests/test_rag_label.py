from web.rag import build_index, retrieve, RAG_HEADER, RAG_FOOTER


def test_rag_context_is_labeled_as_library_not_user_material():
    # Corpus di più testi: con BM25 un termine presente in metà dei documenti ha peso zero
    chunks = ["il montaggio di Bresson e il modello", "la struttura in tre atti secondo Egri",
              "il dialogo come azione", "il personaggio e il desiderio", "lo slow cinema di Lav Diaz"]
    sources = [f"doc{i}.txt" for i in range(len(chunks))]
    ctx = retrieve("Bresson montaggio modello", build_index(chunks), chunks, sources)
    assert ctx.startswith(RAG_HEADER)
    assert ctx.endswith(RAG_FOOTER)
    assert "NON materiali caricati dall'utente" in ctx
