from app.ingestion.chunker import Chunker

def test_chunker_basic():
    chunker = Chunker(chunk_size=10, overlap=2)
    text = "0123456789ABCDEF"
    chunks = chunker.chunk_text(text, "test.txt")
    
    assert len(chunks) > 0
    assert chunks[0].text == "0123456789"
    # overlap is 2, so next starts at 8
    assert chunks[1].text == "89ABCDEF"
    assert chunks[0].metadata.filename == "test.txt"
