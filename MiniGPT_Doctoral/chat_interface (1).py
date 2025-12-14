
#!/usr/bin/env python3
"""
MiniGPT Instructivo - Chat Interface
Proyecto Final de Doctorado en IA/NLP

Uso:
    python chat_interface.py --mode cli      # Interfaz de línea de comandos
    python chat_interface.py --mode gradio   # Interfaz web Gradio
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# Configuración
MODEL_PATH = "./miniGPT_final"  # Ajustar según ubicación
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
print(f"🔄 Cargando modelo desde {MODEL_PATH}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)
model.eval()
print(f"✅ Modelo cargado en {DEVICE}")


def generate_response(
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Genera respuesta para una instrucción."""

    if input_text:
        prompt = f"""### Instrucción:
{instruction}

### Entrada:
{input_text}

### Respuesta:
"""
    else:
        prompt = f"""### Instrucción:
{instruction}

### Respuesta:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Respuesta:" in response:
        response = response.split("### Respuesta:")[-1].strip()
    if "### Instrucción:" in response:
        response = response.split("### Instrucción:")[0].strip()

    return response


def run_cli():
    """Ejecuta interfaz CLI."""
    print("\n" + "="*60)
    print("🤖 MINIGPT INSTRUCTIVO - Chat")
    print("   Escribe 'salir' para terminar")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\n👤 Tú: ").strip()

            if user_input.lower() == "salir":
                print("\n👋 ¡Hasta luego!")
                break

            if not user_input:
                continue

            response = generate_response(user_input)
            print(f"\n🤖 MiniGPT: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break


def run_gradio():
    """Ejecuta interfaz Gradio."""
    import gradio as gr

    def chat_fn(message, history):
        return generate_response(message)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="🤖 MiniGPT Instructivo",
        description="Modelo GPT de 110M parámetros entrenado en español",
        examples=["¿Qué es Python?", "Explica la fotosíntesis", "Escribe un poema corto"],
    )

    demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT Chat Interface")
    parser.add_argument("--mode", choices=["cli", "gradio"], default="cli",
                        help="Modo de interfaz: cli o gradio")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        run_gradio()
