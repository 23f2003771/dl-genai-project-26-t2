import modal

app = modal.App("dlgenai-model")

rag_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "gradio", "langchain", "langchain-community", "torch", 
        "langchain-huggingface", "faiss-cpu", "transformers", "fastapi", 
        "accelerate", "bitsandbytes", "huggingface_hub", "sentence-transformers"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_commands(
        "pip install hf-transfer",
        "hf download BAAI/bge-small-en-v1.5",
        "hf download google/gemma-4-12B",
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
    .add_local_dir(
        "./data/faiss_index",
        remote_path="/faiss_index"
    )
)

@app.cls(
    image=rag_image, 
    gpu="L4", 
    timeout=600, 
    secrets=[modal.Secret.from_name("huggingface-secret")], 
    enable_memory_snapshot=True, 
    experimental_options={"enable_gpu_snapshot": True}
)
@modal.concurrent(max_inputs=100)
class RAGApp:
    
    @modal.enter(snap=True)
    def load_models(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        print("Loading Vector DB and LLM into VRAM...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- VECTOR DB SETUP ---
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", 
            model_kwargs={"device": device, "model_kwargs": {"use_safetensors": False}},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_db = FAISS.load_local(
            "/faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # --- LLM SETUP ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model_id = "google/gemma-4-12B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            local_files_only=True
        )
        
        self.llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer, max_new_tokens=150, do_sample=True, temperature=0.1)

    @modal.asgi_app()
    def ui(self):
        import gradio as gr
        from fastapi import FastAPI
        from gradio.routes import mount_gradio_app

        def answer_question(question):
            wiki_docs = self.vector_db.similarity_search(question, k=3)
            wiki_context = "\n".join([d.page_content for d in wiki_docs])
            
            prompt = (
                "Instruction: You are an expert science assistant. Read the provided Context and answer the Question. "
                "Provide a direct and concise answer.\n\n"
                f"Context:\n{wiki_context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:\n"
            )
            
            raw_output = self.llm_pipe(prompt)[0]["generated_text"]
            answer = raw_output.replace(prompt, "").strip() 
            return answer

        # --- GRADIO UI ---
        interface = gr.Interface(
            fn=answer_question,
            inputs=gr.Textbox(lines=15, placeholder="Ask a science question on the knowledge base..."),
            outputs=gr.Textbox(lines=15, max_lines=30),
            title="Physics & Science Q&A RAG",
            description="Powered by Modal (Scale-to-Zero L4 GPU) & Gemma 4-12B.",
            flagging_mode="never"
        )
        
        fastapi_app = FastAPI()
        return mount_gradio_app(
            app=fastapi_app,
            blocks=interface,
            path="/"
        )