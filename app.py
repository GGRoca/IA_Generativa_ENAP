import os
import streamlit as st
from typing import List, Dict, Any
import time
import re

from dotenv import load_dotenv

# LangChain e submódulos
#from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema import BaseRetriever

def init_session_state():
    """Inicializa variáveis de estado da sessão do Streamlit"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    if "chain" not in st.session_state:
        st.session_state.chain = None


def extract_metadata_from_content(doc: Document) -> Document:
    """Extrai metadados do conteúdo do documento para enriquecê-lo"""
    content = doc.page_content
    
    # Melhorar a extração de IN com regex mais abrangente
    in_match = re.search(r'IN\s+n[º°\.]\s*(\d+)', content, re.IGNORECASE) or \
               re.search(r'Instru[çc][aã]o\s+Normativa\s+n[º°\.]\s*(\d+)', content, re.IGNORECASE) or \
               re.search(r'Instru[çc][aã]o\s+Normativa\s+(\d+)', content, re.IGNORECASE)
    
    # Extrair menções a artigos com regex mais amplo
    art_matches = re.findall(r'Art[\.\s]*\s*(\d+)', content, re.IGNORECASE) or \
                  re.findall(r'Artigo\s+(\d+)', content, re.IGNORECASE)
    
    # Extrair incisos com numeração romana
    inciso_matches = re.findall(r'(?:inciso|item)\s+([IVXLCDM]+)', content, re.IGNORECASE)
    
    # Atualizar metadados
    if in_match:
        doc.metadata['instrucao_normativa'] = f"IN {in_match.group(1)}"
    
    if art_matches:
        doc.metadata['artigos'] = [f"Art. {art}" for art in art_matches]
        
    if inciso_matches:
        doc.metadata['incisos'] = [f"inciso {inciso}" for inciso in inciso_matches]
    
    return doc


def enrich_documents(docs: List[Document]) -> List[Document]:
    """Enriquece documentos com metadados extraídos e formatação"""
    enriched_docs = []
    
    for i, doc in enumerate(docs):
        # Extrair metadados do conteúdo
        doc = extract_metadata_from_content(doc)
        
        # Adicionar informações de fonte para facilitar citações
        source_info = ""
        if 'instrucao_normativa' in doc.metadata:
            source_info += f"{doc.metadata['instrucao_normativa']} - "
        if 'artigos' in doc.metadata and doc.metadata['artigos']:
            source_info += f"{', '.join(doc.metadata['artigos'])} - "
        if 'source' in doc.metadata:
            filename = os.path.basename(doc.metadata['source'])
            source_info += f"Fonte: {filename}"
        
        # Formatar conteúdo para melhor leitura pelo LLM
        formatted_content = f"""
TRECHO {i+1}:
{source_info}
---
{doc.page_content}
---
"""
        doc.page_content = formatted_content
        enriched_docs.append(doc)
    
    return enriched_docs


def create_retriever():
    """Cria o retriever melhorado para busca de documentos"""
    try:
        # Configuração do modelo de embeddings
        embedding_engine = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Carrega a base vetorial
        db_pasta = "db_vetorial"
        #vector_db = Chroma(
            #persist_directory=db_pasta,
            #embedding_function=embedding_engine
        vector_db = FAISS.from_documents(text_chunks, embedding_engine)
        )

        # Função de filtragem personalizada
        def filter_by_metadata_richness(docs):
            """Filtra documentos com base na riqueza de metadados"""
            scored_docs = []
            for doc in docs:
                score = 0
                if 'instrucao_normativa' in doc.metadata:
                    score += 3  # Prioridade para documentos com IN
                if 'artigos' in doc.metadata:
                    score += len(doc.metadata['artigos'])
                if 'source' in doc.metadata:
                    score += 1  # Pontuação básica para documentos com fonte
                
                scored_docs.append((doc, score))
            
            # Seleciona os melhores documentos
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:10]]  # Top 10 documentos

        # Configuração do retriever base com MMR
        base_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30
            }
        )

        # Implementação do Custom Retriever
        class CustomRetriever(BaseRetriever):
            """Retriever personalizado que combina MMR com filtragem pós-processamento"""
            
            def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
                # Primeiro obtém os documentos do retriever base
                base_docs = base_retriever.get_relevant_documents(query)
                # Depois aplica o filtro personalizado
                return filter_by_metadata_richness(base_docs)

            async def _aget_relevant_documents(self, query: str, *, run_manager):
                # Implementação assíncrona se necessário
                return self._get_relevant_documents(query)

        # Cria e retorna o retriever personalizado
        return CustomRetriever()
        
    except Exception as e:
        st.error(f"Erro ao criar o retriever: {str(e)}")
        return None


def create_chain():
    """Cria a chain RAG com configurações personalizadas"""
    try:
        # Carrega variáveis de ambiente
        #load_dotenv()
        #openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        
        if not openai_api_key:
            st.error("API Key da OpenAI não encontrada. Configure-a no arquivo .env")
            return None
        
        # Cria o retriever
        retriever = create_retriever()
        if not retriever:
            return None
        
        # Define o prompt personalizado com instruções mais flexíveis para o LLM
        system_template = """\
Você é um assistente especializado em legislação da ANCINE (Agência Nacional do Cinema), responsável por fornecer informações precisas sobre instruções normativas.

--- Conteúdo relevante da base de dados sobre INs da ANCINE ---
{context}

---- INSTRUÇÕES CRÍTICAS ----
VOCÊ DEVE SEMPRE CITAR EXPLICITAMENTE OS NÚMEROS DAS INs QUANDO DISPONÍVEIS. Esta é uma EXIGÊNCIA ABSOLUTA.

1) SEMPRE comece sua resposta identificando em qual IN a informação foi encontrada. Use exatamente a numeração que aparece nos trechos.
   Exemplo correto: "De acordo com a IN nº 95, ..."
   Exemplo incorreto: "Em uma das INs da ANCINE, consta que..."

2) Se você identificar o conteúdo mas não encontrar o número da IN nos trechos fornecidos, PROCURE NOS METADADOS:
   - Procure por tags como [instrucao_normativa: IN XX] ou frases que indiquem o documento de origem
   - Verifique o nome do arquivo na referência (geralmente contém o número da IN)

3) Se você ainda não conseguir determinar o número da IN, então e SOMENTE então, use "Segundo regulamentação da ANCINE" e continue com informações precisas.

4) SEMPRE inclua citações específicas a artigos, parágrafos e incisos quando presentes.

5) NUNCA responda de forma genérica quando informações específicas estiverem disponíveis.

Este sistema é uma ferramenta de consulta legal e DEVE fornecer referências precisas. Respostas sem citações específicas são consideradas falhas.

# Adicionar ao seu prompt:

Técnicas que você DEVE usar para responder corretamente:

1) EXAMINE TODOS OS TRECHOS FORNECIDOS em busca de menções a números de INs. Não se limite aos primeiros trechos.

2) EXAMINE OS METADADOS de cada documento. Procure por padrões como:
   - [instrucao_normativa: IN XX]
   - Nomes de arquivos como "IN_95_texto.pdf" 
   - Referências explícitas no texto como "Instrução Normativa nº XX"

3) Se o mesmo conteúdo aparecer em múltiplos trechos, VERIFIQUE se algum deles contém o número da IN.

4) Quando mencionar artigos ou incisos, SEMPRE indique a qual IN pertence.

5) Prefira ser específico mesmo que só tenha certeza parcial. É melhor dizer "A informação parece constar na IN 95, conforme indicado no texto" do que omitir a referência.

EXEMPLOS DE RESPOSTAS INACEITÁVEIS:
- "Em uma das INs da ANCINE, consta que..."
- "Segundo as instruções normativas da ANCINE..."
- "Para entender as diferenças... é importante considerar as definições e requisitos estabelecidos nas instruções normativas da ANCINE."

EXEMPLO DE RESPOSTA ACEITÁVEL:
"De acordo com a IN nº 95, Art. 2º, inciso VII, uma obra audiovisual brasileira é definida como..."



Agora, responda à pergunta do usuário com todas as referências específicas que conseguir extrair dos trechos:
{question}
"""
        
        prompt_template = PromptTemplate(
            template=system_template,
            input_variables=["context", "question"]
        )
        
        # Configura o modelo LLM
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="o3-mini",
            #temperature=0.1,
            max_tokens=6000,
            request_timeout=90,
        )

        # Cria a chain com memória e prompt customizado
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt_template},
            return_source_documents=True,
            output_key="answer",
            verbose=True,
        )
        
        # Função para processar a query antes de enviar ao retriever
        def process_query(query):
            # Expandir a query para melhorar a recuperação
            expansions = {
                "cadastrar": ["cadastro", "registro", "registrar", "inscrição", "inscrever"],
                "empresa": ["empresa", "produtora", "distribuidor", "distribuidora", "companhia", "pessoa jurídica"],
                "publicidade": ["publicidade", "propaganda", "anúncio", "comercial", "intervalo comercial"],
                "limite": ["limite", "limitação", "tempo", "duração", "restrição"],
                "SCB": ["Sistema de Controle de Bilheteria"],
                "SADIS": ["Sistema de Acompanhamento de Distribuição em Salas"],
                "SAVI": ["Sistema de Acompanhamento de Video Doméstico"],
                "SRPTV": ["Sistema de Recepção de Programação de TV Paga"],
                "SAD": ["Sistema Ancine Digital"],
                "CPB": ["Certificado de Produto Brasileiro"],
                "ROE": ["Registro de Obra Estrangeira"],
                "CRT": ["Certificado de Registro de Título"]
            }
            
            expanded_query = query
            for key, alternatives in expansions.items():
                if key.lower() in query.lower():
                    for alt in alternatives:
                        if alt.lower() not in query.lower():
                            expanded_query += f" {alt}"
            
            return expanded_query
        
        # Criamos um wrapper para a chain original que processará os documentos
        def chain_with_preprocessing(inputs):
            # Processar a query para melhorar a recuperação
            question = inputs["question"]
            expanded_question = process_query(question)
            
            # NOVO CÓDIGO ADICIONADO AQUI
            # Recuperar documentos relevantes
            docs = retriever.get_relevant_documents(expanded_question)
            
            # Fazer análise preliminar das INs e artigos
            ins_mentioned = {}
            for doc in docs:
                if 'instrucao_normativa' in doc.metadata:
                    in_num = doc.metadata['instrucao_normativa']
                    if in_num not in ins_mentioned:
                        ins_mentioned[in_num] = []
                        
                    # Extrair artigos mencionados no conteúdo
                    art_pattern = r'Art[\.\s]*\s*(\d+)'
                    art_matches = re.findall(art_pattern, doc.page_content, re.IGNORECASE)
                    ins_mentioned[in_num].extend([f"Art. {m}" for m in art_matches])
            
            # Criar resumo contextual
            context_summary = "\n\nINFORMAÇÕES DE CONTEXTO IMPORTANTES:\n"
            for in_num, articles in ins_mentioned.items():
                unique_articles = list(set(articles))
                context_summary += f"- {in_num} contém os seguintes artigos relevantes: {', '.join(unique_articles[:10])}\n"
            
            # Combinar com a query expandida
            final_question = f"{expanded_question}\n\n{context_summary}"
            # FIM DO NOVO CÓDIGO
            
            # Executar a chain com a query final aprimorada
            result = chain({"question": final_question})
            
            # Processar os documentos de origem antes de retornar
            if "source_documents" in result:
                result["source_documents"] = enrich_documents(result["source_documents"])
                
            return result
        
        # Retornamos a função wrapped em vez da chain direta
        return chain_with_preprocessing
        
    except Exception as e:
        st.error(f"Erro ao criar a chain: {str(e)}")
        return None


def display_chat_history():
    """Exibe o histórico de conversa na interface"""
    for message in st.session_state.messages:
        if message["role"] == "human":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])


def extract_reference_summary(docs):
    """Extrai um resumo das referências encontradas para mostrar ao usuário"""
    if not docs:
        return "Nenhuma fonte específica encontrada."
    
    references = []
    ins_mentioned = set()
    articles_mentioned = set()
    
    for doc in docs:
        # Extrai IN
        if 'instrucao_normativa' in doc.metadata:
            ins_mentioned.add(doc.metadata['instrucao_normativa'])
        
        # Extrai artigos mencionados
        if 'artigos' in doc.metadata and doc.metadata['artigos']:
            for art in doc.metadata['artigos']:
                articles_mentioned.add(art)
        
        # Extrai fonte do documento
        if 'source' in doc.metadata:
            filename = os.path.basename(doc.metadata['source'])
            references.append(filename)
    
    # Formata resumo
    summary = []
    if articles_mentioned:
        summary.append(f"**Artigos mencionados:** {', '.join(sorted(articles_mentioned))}")
    
    if ins_mentioned:
        summary.append(f"**Instruções Normativas:** {', '.join(sorted(ins_mentioned))}")
    
    unique_refs = list(set(references))
    if unique_refs:
        summary.append(f"**Documentos consultados:** {', '.join(unique_refs[:3])}")
    
    return "\n".join(summary)


def main():
    # Configuração da página
    st.set_page_config(
        page_title="Chatbot da ANCINE",
        page_icon="🎬",
        layout="wide"
    )
    
    # Inicializa variáveis de estado
    init_session_state()
    
    # Interface principal
    st.title("🎬 Chatbot de Instruções Normativas da ANCINE")
    st.markdown("""
    Tire suas dúvidas sobre instruções normativas e regulamentações do segmento de regulação e registro do mercado audiovisual brasileiro .
    Este chatbot consulta a base de documentos oficiais da ANCINE para fornecer informações precisas.
    """)
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reiniciar Chat", use_container_width=True):
                st.session_state.chain = None
                st.success("Chatbot reiniciado.")
        
        with col2:
            if st.button("🗑️ Limpar Histórico", use_container_width=True):
                st.session_state.messages = []
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                st.session_state.chain = None
                st.rerun()  # Substituído experimental_rerun por rerun
        
        st.markdown("---")
        st.markdown("""
        ### Exemplos de perguntas:
        - O que é uma obra brasileira independente?
        - Quais os requisitos para cadastro de obra?
        - Como faço para registrar um projeto na ANCINE?
        - O que diz a IN 95 sobre produção?
        - Classificação indicativa segundo a ANCINE
        """)
        
        st.markdown("---")
        st.markdown("Desenvolvido com ❤️ usando LangChain e Streamlit para a ENAP")
    
    # Inicializa a chain se ainda não existir
    if st.session_state.chain is None:
        with st.spinner("Inicializando o chatbot especialista em ANCINE..."):
            st.session_state.chain = create_chain()
            if st.session_state.chain is None:
                st.error("Não foi possível inicializar o chatbot. Verifique sua API Key no arquivo .env")
                return
    
    # Exibe histórico de mensagens
    display_chat_history()
    
    # Campo de entrada do usuário
    user_query = st.chat_input("Digite sua pergunta sobre instruções normativas da ANCINE...")
    
    # Processamento da pergunta
    if user_query:
        # Adiciona a pergunta ao histórico
        st.session_state.messages.append({"role": "human", "content": user_query})
        
        # Exibe a pergunta
        with st.chat_message("user"):
            st.write(user_query)
        
        # Exibe indicador de "pensando"
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Consultando a base de INs da ANCINE...")
            
            try:
                # Executa a chain para obter a resposta
                start_time = time.time()
                response = st.session_state.chain({"question": user_query})
                end_time = time.time()
                
                # Formata a resposta
                answer = response.get("answer", "Desculpe, não consegui encontrar informações sobre este tema nas INs consultadas.")
                
                # Análise e resumo das fontes
                source_docs = response.get("source_documents", [])
                if source_docs:
                    reference_summary = extract_reference_summary(source_docs)
                    answer += f"\n\n---\n{reference_summary}"
                
                # Adiciona tempo de resposta
                answer += f"\n\n*Tempo de resposta: {end_time - start_time:.2f} segundos*"
                
                # Atualiza o placeholder com a resposta
                message_placeholder.markdown(answer)
                
                # Salva a resposta no histórico
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Erro ao processar sua pergunta: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()