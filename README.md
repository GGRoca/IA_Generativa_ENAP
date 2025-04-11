# 🤖 Chatbot de Instruções Normativas de Regulação da ANCINE

[![Licença MIT](https://img.shields.io/badge/Licença-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status do Projeto](https://img.shields.io/badge/Status-Ativo-brightgreen.svg)](https://github.com/GGRoca/IA_Generativa_ENAP)

Solução de IA para consulta de instruções normativas e regulamentações da ANCINE utilizando técnicas avançadas de NLP.

## 🌟 Funcionalidades Principais

- 🧠 Busca semântica em Instruções Normativas regulatórias
- 📑 Citação automática de INs e artigos relevantes
- 🔍 Filtragem inteligente por metadados
- 💬 Interface conversacional natural
- 📊 Análise de contexto automatizada

## 🔗📽️ Link para acessar o app
 - https://iagenerativaenap-a3eu9mrcjys9jcrlqufxtx.streamlit.app/


## 📦 Pré-requisitos

- Python 3.10+
- OpenAI API Key
- Git

## 🚀 Instalação Rápida

# Clonar repositório
git clone https://github.com/GGRoca/IA_Generativa_ENAP.git

# Acessar diretório
cd ancine-chatbot

# Instalar dependências
pip install -r requirements.txt


⚙️ Configuração
1. Crie um arquivo .env na raiz do projeto:

OPENAI_API_KEY=sua-chave-aqui
DB_VETORIAL_PATH=db_vetorial

2. Execute o Streamlit:

streamlit run app.py


🖥️ Uso
# Exemplo de consulta
pergunta = "Quais os requisitos para cadastro de obra brasileira?"
resposta = chatbot.consultar(pergunta)
print(resposta)
🛠️ Tecnologias Utilizadas
NLP: OpenAI o3-mini

Vector DB: FAISS

Embeddings: HuggingFace (all-mpnet-base-v2)

Framework: LangChain

Interface: Streamlit

📌 Exemplos de Consultas
"Como registrar uma obra audiovisual?"

"Quais as exigências da IN 95 para produções independentes?"

"Explique o artigo 12 da IN 128 sobre distribuição"

"Qual o prazo de envio de informações para o SADIS?"

🤝 Como Contribuir
Faça um Fork do projeto

Crie sua Branch (git checkout -b feature/nova-funcionalidade)

Commit suas mudanças (git commit -m 'Adiciona nova funcionalidade')

Push para a Branch (git push origin feature/nova-funcionalidade)

Abra um Pull Request

📄 Licença
Distribuído sob licença MIT.


⚠️ Nota Importante: Este projeto é mantido é um trabalho acadêmico sem fins institucionais. Não é um sistema oficial da ANCINE.





