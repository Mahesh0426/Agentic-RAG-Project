from typing import List

import logfire


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        # Return early if input text is empty or only whitespace
        if not text.strip(): 
            return []
            
        # Split text into paragraphs using double newline as natural boundaries
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            # If adding the next paragraph stays within chunk_size, accumulate it
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                # Chunk size reached: save current chunk and start a new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        
        # Append the final remaining chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        # Filter out any accidental blank chunks and log the outcome
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks