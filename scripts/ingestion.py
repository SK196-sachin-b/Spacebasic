import os
from pathlib import Path
import pdfplumber

from langchain_aws import BedrockEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db import db
from embedding import embedding_service


class DocumentIngestion:
    def __init__(self):

        # 🔹 Embeddings (for semantic chunking only)
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0"
        )

        # 🔹 Semantic chunkers
        self.semantic_chunker_policy = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )

        self.semantic_chunker_manual = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75
        )

        # 🔹 FAQ splitter
        self.faq_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )

    # ──────────────────────────────────────────────
    # READ PDF
    # ──────────────────────────────────────────────
    def read_pdf(self, pdf_path):
        """
        Extract text page-wise
        Returns: [(page_number, text)]
        """
        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append((page_num, text.strip()))

            print(f"   📖 {len(pages)} pages extracted")

        except Exception as e:
            print(f"❌ Failed to read {pdf_path.name}: {e}")

        return pages

    # ──────────────────────────────────────────────
    # FAQ CHUNKING
    # ──────────────────────────────────────────────
    def chunk_faq(self, text):
        if not text or not text.strip():
            return []
        chunks = self.faq_splitter.split_text(text)
        print(f"   ✅ FAQ: {len(chunks)} chunks")
        return chunks

    # ──────────────────────────────────────────────
    # SEMANTIC CHUNKING
    # ──────────────────────────────────────────────
    def semantic_chunk(self, text, chunker, label):
        if not text or not text.strip():
            return []

        if len(text) < 100:
            return [text]

        try:
            chunks = chunker.split_text(text)

            if not chunks:
                return [text]

            # Merge small chunks
            merged = []
            buffer = ""

            for chunk in chunks:
                if len(chunk) < 100:
                    buffer += " " + chunk
                else:
                    if buffer:
                        merged.append(buffer.strip())
                        buffer = ""
                    merged.append(chunk)

            if buffer:
                merged.append(buffer.strip())

            print(f"   ✅ {label}: {len(merged)} chunks")
            return merged

        except Exception as e:
            print(f"   ❌ {label} failed: {e}")
            return [text]

    # ──────────────────────────────────────────────
    # ROUTER
    # ──────────────────────────────────────────────
    def chunk_text(self, text, folder):
        folder = folder.lower().strip()

        print(f"\n   📂 Chunking strategy: [{folder}]")

        if folder == "faq":
            return self.chunk_faq(text)

        elif folder == "policy":
            return self.semantic_chunk(
                text,
                self.semantic_chunker_policy,
                "Policy"
            )

        elif folder == "user_manuals":
            return self.semantic_chunk(
                text,
                self.semantic_chunker_manual,
                "Manual"
            )

        else:
            print("   ⚠️ Unknown folder → using default")
            return self.semantic_chunk(
                text,
                self.semantic_chunker_policy,
                "Default"
            )

    # ──────────────────────────────────────────────
    # INGEST ALL PDFs
    # ──────────────────────────────────────────────
    def ingest_documents(self):

        print("\n" + "="*60)
        print("📚 DOCUMENT INGESTION STARTED")
        print("="*60)

        if not db.connect():
            return False

        if not db.create_table():
            return False

        base_path = Path("data")
        if not base_path.exists():
            base_path = Path("../data")

        pdf_files = list(base_path.rglob("*.pdf"))

        if not pdf_files:
            print("❌ No PDFs found")
            return False

        print(f"\n📄 Found {len(pdf_files)} PDFs")

        total_chunks = 0

        for pdf_path in pdf_files:

            print(f"\n{'='*50}")
            print(f"📄 {pdf_path.parent.name}/{pdf_path.name}")
            print(f"{'='*50}")

            folder = pdf_path.parent.name
            file_name = pdf_path.name

            # 🔥 AUTO VERSION
            version = db.get_next_version(file_name)

            pages = self.read_pdf(pdf_path)
            if not pages:
                continue

            pdf_chunks = 0

            for page_num, page_text in pages:

                print(f"\n   📃 Page {page_num}")

                chunks = self.chunk_text(page_text, folder)

                for chunk in chunks:

                    if not chunk.strip():
                        continue

                    embedding = embedding_service.embed_text(chunk)
                    if embedding is None:
                        continue

                    db.insert_document(
                        content=chunk,
                        embedding=list(embedding),
                        source_file=file_name,
                        folder=folder,
                        page_number=page_num,
                        version=version   # ✅ IMPORTANT
                    )

                    total_chunks += 1
                    pdf_chunks += 1

            print(f"   ✅ Stored {pdf_chunks} chunks (v{version})")

        # 🔹 Create vector index AFTER ingestion
        print("\n🔄 Creating vector index...")
        db.create_index_after_ingestion()

        # 🔥 Reload BM25 corpus after ingestion
        print("🔄 Reloading BM25 corpus...")
        from retrieval import retrieval_service
        retrieval_service.reload_bm25()

        db.close()

        print("\n" + "="*60)
        print("📊 INGESTION SUMMARY")
        print("="*60)
        print(f"✅ Total chunks: {total_chunks}")
        print("="*60)

        return True

    # ──────────────────────────────────────────────
    # INGEST SINGLE PDF
    # ──────────────────────────────────────────────
    def ingest_single_pdf(self, file_path, folder=None):

        if not db.connect():
            return False

        # ✅ FIX 1: define file_name
        file_name = os.path.basename(file_path)

        print(f"\n📄 Processing: {file_name}")

        # ✅ FIX 2: auto detect folder (faq/policy/manual)
        if not folder:
            folder = os.path.basename(os.path.dirname(file_path)).lower()

        print(f"📂 Detected folder: {folder}")

        # ✅ FIX 3: version
        version = db.get_next_version(file_name)

        pages = self.read_pdf(file_path)
        if not pages:
            return False

        total_chunks = 0

        for page_num, page_text in pages:

            chunks = self.chunk_text(page_text, folder)

            for chunk in chunks:

                if not chunk.strip():
                    continue

                embedding = embedding_service.embed_text(chunk)
                if embedding is None:
                    continue

                db.insert_document(
                    content=chunk,
                    embedding=list(embedding),
                    source_file=file_name,
                    folder=folder,
                    page_number=page_num,
                    version=version
                )

                total_chunks += 1

        print(f"✅ Stored {total_chunks} chunks (v{version})")

        # ✅ FIX 4: index creation
        db.create_index_after_ingestion()
        
        # 🔥 Reload BM25 corpus after ingestion
        print("🔄 Reloading BM25 corpus...")
        from retrieval import retrieval_service
        retrieval_service.reload_bm25()
        
        db.close()

        return True


# Global instance
ingestion_service = DocumentIngestion()