# Este arquivo cria o banco vetorial e persiste no disco.
# Você poderá chamá-lo uma única vez sempre que adicionar/alterar PDFs.

import os
import re
from dotenv import load_dotenv

# Ferramentas para lidar com PDFs e dividir textos
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Para embeddings e banco vetorial
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def criar_base_vetorial():
    """
    Função que:
    1) Carrega PDFs de 'INs_Ancine_PDFs'
    2) Separa em chunks
    3) Gera embeddings e cria base Chroma persistida em 'db_vetorial'
    """

    # 1. Carrega as variáveis de ambiente
    load_dotenv()  # Lê o arquivo .env no diretório atual ou acima
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("Aviso: OPENAI_API_KEY não foi encontrada em .env")

    # 2. Carrega PDFs
    pasta_pdfs = "INs_Ancine_PDFs"
    pdf_files = [
        os.path.join(pasta_pdfs, f)
        for f in os.listdir(pasta_pdfs)
        if f.lower().endswith(".pdf")
    ]

    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Total de PDFs encontrados: {len(pdf_files)}")
    print(f"Total de documentos carregados: {len(all_docs)}")

    # 3. Divide em chunks
    chunk_size = 1500
    chunk_overlap = 300
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Priorizar quebras de parágrafo
    )
    
    # Extrair e armazenar metadados ANTES do chunking
    docs_with_metadata = []
    for doc in all_docs:
        # Extrair número da IN do título do documento ou conteúdo
        in_match = re.search(r'IN\s+n[º°\.]\s*(\d+)', doc.page_content, re.IGNORECASE) or \
                  re.search(r'Instru[çc][aã]o\s+Normativa\s+n[º°\.]\s*(\d+)', doc.page_content, re.IGNORECASE) or \
                  re.search(r'Instru[çc][aã]o\s+Normativa\s+(\d+)', doc.page_content, re.IGNORECASE)
        
        if in_match:
            doc.metadata['instrucao_normativa'] = f"IN {in_match.group(1)}"
        
        # Também pode usar o nome do arquivo como referência
        if 'source' in doc.metadata:
            filename = os.path.basename(doc.metadata['source'])
            in_file_match = re.search(r'IN[-_\s]*(\d+)', filename, re.IGNORECASE)
            if in_file_match and 'instrucao_normativa' not in doc.metadata:
                doc.metadata['instrucao_normativa'] = f"IN {in_file_match.group(1)}"
        
        docs_with_metadata.append(doc)
    
    # Agora realize o chunking preservando os metadados
    text_chunks = text_splitter.split_documents(docs_with_metadata)

    print(f"Total de chunks gerados: {len(text_chunks)}")

    # 4. Cria embeddings e base vetorial (Chroma)
    embedding_engine = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}  # ou 'cuda' se estiver configurado
    )

    db_pasta = "db_vetorial"  # Pasta de persistência do Chroma
    vector_db = FAISS.from_documents(text_chunks, embedding_engine)
    vector_db.save_local("db_vetorial")  # persiste os arquivos, como index.faiss
    # Em versões Chroma >= 0.4.x, persist() é automático
    # vector_db.persist()  # Se precisar, dependendo da sua versão
    print("Base vetorial criada.")

    # 5. Exibir tamanho em MB
    # Chroma hoje costuma chamar o arquivo principal de "chroma.sqlite3" ou algo similar
    # Então vou checar os arquivos .sqlite3 que encontrei na pasta
    tamanho_total = 0.0
    for f in os.listdir(db_pasta):
        if f.endswith(".sqlite") or f.endswith(".sqlite3"):
            full_path = os.path.join(db_pasta, f)
            tamanho_total += os.path.getsize(full_path)

    tamanho_mb = tamanho_total / (1024 * 1024)  # converte bytes -> MB
    print(f"Tamanho total da base: {tamanho_mb:.2f} MB")


if __name__ == "__main__":
    criar_base_vetorial()
