"""
CHATBOT DETECTOR DE PLANTAS - UNIVERSIDAD DE CUNDINAMARCA
Versión: GPT-OSS-20B (OpenRouter) + YOLOv8 Detector
"""

import os
import json
import csv
import re
import time
import gradio as gr
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from detector import DetectorDePlantas
from openai import OpenAI
import base64


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
MODELO_PATH = r"C:\Users\julia\Downloads\Red Neuronal YOLO - copia\runs\detect\especies_colombianas\weights/best.pt"
CONF_MIN = 0.50
DIRECTORIO_TEMPORAL = "tmp"
DIRECTORIO_REPORTES = "reportes"
ARCHIVO_RESPUESTAS = r"C:\Users\julia\Downloads\Red Neuronal YOLO - copia\Red Neuronal YOLO - copia/answers.json"
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "logo.jpeg")
# Configuración de OpenRouter
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "gpt-oss-20b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ==========================================================
# INICIALIZACIÓN
# ==========================================================
detector = DetectorDePlantas(modelo_path=MODELO_PATH, confianza_minima=CONF_MIN)
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

qa_database = {}
try:
    with open(ARCHIVO_RESPUESTAS, 'r', encoding='utf-8') as f:
        qa_database = json.load(f)
except Exception as e:
    print(f"Error cargando respuestas JSON: {e}")

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def _crear_directorios():
    for directorio in [DIRECTORIO_TEMPORAL, DIRECTORIO_REPORTES]:
        Path(directorio).mkdir(exist_ok=True)

def _guardar_imagen_temporal(imagen_pil) -> str:
    _crear_directorios()
    nombre = f"imagen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    ruta = os.path.join(DIRECTORIO_TEMPORAL, nombre)
    imagen_pil.save(ruta)
    if not os.path.exists(ruta):
        raise FileNotFoundError("La imagen temporal no se guardó correctamente.")
    return ruta

def _normalizar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    tildes = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}
    for tilde, sin_tilde in tildes.items():
        texto = texto.replace(tilde, sin_tilde)
    return re.sub(r'\s+', ' ', texto)

def _parse_csv_detecciones(csv_path: str) -> Dict[str, any]:
    stats = {'total_detecciones': 0, 'especies': {}, 'confianza_promedio': 0.0, 'confianzas': [], 'detalles': []}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                especie = row.get('Clase Nombre', row.get('class', 'desconocida')).strip()
                conf = float(row.get('Confianza', row.get('confidence', 0)))
                stats['total_detecciones'] += 1
                stats['confianzas'].append(conf)
                if especie not in stats['especies']:
                    stats['especies'][especie] = {'count': 0, 'confianzas': []}
                stats['especies'][especie]['count'] += 1
                stats['especies'][especie]['confianzas'].append(conf)
                stats['detalles'].append({'especie': especie, 'confianza': conf})
        if stats['confianzas']:
            stats['confianza_promedio'] = sum(stats['confianzas']) / len(stats['confianzas'])
    except Exception as e:
        print(f"Error parseando CSV: {e}")
    return stats

def _normalizar_especie_base(nombre_clase: str) -> str:
    nombre_norm = nombre_clase.lower()
    partes_morf = ["hoja", "fruto", "flor", "frutov", "frutom", "vaina"]
    for sufijo in partes_morf:
        if nombre_norm.endswith(f"_{sufijo}"):
            return nombre_clase[:-(len(sufijo) + 1)]
    return nombre_clase

def _formatear_analisis_completo(especie_detectada: str, stats: Dict) -> str:
    especie_norm = _normalizar_texto(especie_detectada)
    especie_base = especie_norm.split("_hoja")[0].split("_flor")[0].split("_fruto")[0]
    species_data = qa_database.get("species_data", {})
    datos = None
    for clave in species_data.keys():
        clave_norm = _normalizar_texto(clave)
        if especie_base in clave_norm or clave_norm in especie_base:
            datos = species_data[clave]
            especie_base = clave
            break
    if not datos:
        gpt_info = _consultar_openrouter(
            f"Describe en detalle la especie vegetal '{especie_base}', incluyendo morfología, cultivo, usos, plagas y relevancia económica."
        )
        return f" Detección: {especie_detectada}\nNo se encontraron datos locales. Consulta a GPT:\n{gpt_info}"
    respuesta = f" ANÁLISIS COMPLETO: {especie_base}\n"
    if datos.get("scientific_name"):
        respuesta += f"Nombre científico: {datos['scientific_name']}\n"
    if datos.get("family"):
        respuesta += f" Familia: {datos['family']}\n"
    if datos.get("origin"):
        respuesta += f" Origen: {datos['origin']}\n"
    if datos.get("morphology"):
        respuesta += f"\n MORFOLOGÍA:\n  {datos['morphology']}\n"
    cult = datos.get("cultivation", {})
    if isinstance(cult, dict) and any(cult.values()):
        respuesta += "\n CULTIVO:\n"
        for k, v in cult.items():
            respuesta += f"  • {k.replace('_', ' ').title()}: {v}\n"
    if datos.get("pests_diseases"):
        respuesta += f"\n🐛 PLAGAS Y ENFERMEDADES:\n  {datos['pests_diseases']}\n"
    if datos.get("uses"):
        respuesta += f"\n USOS:\n  {datos['uses']}\n"
    if datos.get("economic_data"):
        respuesta += f"\n DATOS ECONÓMICOS:\n  {datos['economic_data']}\n"
    if stats and 'especies' in stats and especie_detectada in stats['especies']:
        info = stats['especies'][especie_detectada]
        respuesta += "\n DETECCIÓN EN SESIÓN:\n"
        respuesta += f"  • Detectada: {info.get('count', 0)} veces\n"
        respuesta += f"  • Confianza promedio: {info.get('confianza_promedio', 0):.2f}\n"
    return respuesta

def _buscar_respuesta_json(mensaje: str) -> Optional[str]:
    mensaje_norm = _normalizar_texto(mensaje)
    for especie, datos in qa_database.get("species_data", {}).items():
        nombres = [especie.replace("_", " ")] + datos.get("common_names", [])
        for nombre in nombres:
            if _normalizar_texto(nombre) in mensaje_norm:
                return _formatear_analisis_completo(especie, {})
    return None

def _consultar_openrouter(prompt: str) -> str:
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "Eres un bot especializado en botánica y agricultura. Responde con precisión, claridad y rigor científico."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7,
            timeout=20
        )
        if time.time() - start > 20:
            return " Tiempo de respuesta excedido (20 segundos)."
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f" No se pudo conectar con el modelo (error: {e})"

# ==========================================================
# CHATBOT PRINCIPAL
# ==========================================================
def bot_responder(mensaje, historial, imagen, estado):
    msg = (mensaje or "").strip()
    msg_lower = msg.lower()
    if not imagen:
        if msg_lower in ("ayuda", "help", "?"):
            respuesta = " Este chatbot detecta plantas e informa sobre sus características y cultivo. Puedes escribir: '¿Qué sabes del café?' o subir una imagen."
        else:
            respuesta = _buscar_respuesta_json(msg)
            if not respuesta:
                respuesta = _consultar_openrouter(msg)
        historial = historial + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": respuesta}
        ]
        return historial, estado
    try:
        ruta_img = _guardar_imagen_temporal(imagen)
        ruta_csv = detector.detectar(ruta_imagen=ruta_img, carpeta_salida=DIRECTORIO_REPORTES)
        stats = _parse_csv_detecciones(ruta_csv)
        estado["stats"] = stats
        especies = list(stats["especies"].keys())
        if especies:
            especie = especies[0]
            respuesta = _formatear_analisis_completo(especie, stats)
        else:
            respuesta = "No se detectaron plantas en la imagen."
    except Exception as e:
        respuesta = f" Error al analizar la imagen: {e}"
    historial = historial + [
        {"role": "user", "content": msg or "📷 Imagen cargada"},
        {"role": "assistant", "content": respuesta}
    ]
    return historial, estado

# ==========================================================
# INTERFAZ GRADIO (con personalización solicitada)
# ==========================================================
with gr.Blocks(css="""
#navbar {
    background-color: #014023;
    padding: 12px;
    display: flex;
    align-items: center;
    color: white;
    font-size: 20px;
    font-weight: bold;
}
#navbar img {
    height: 45px;
    margin-right: 12px;
}
.gr-button.primary {
    background-color: #014023 !important;
    border-color: #014023 !important;
}
""") as demo:




    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    gr.HTML(f"""
        <div id='navbar' style='display:flex;align-items:center;gap:10px;padding:10px;background-color:#014023;border-radius:10px;'>
            <img src='data:image/jpeg;base64,{logo_base64}' alt='Logo' style='height:50px;border-radius:8px;'>
            <span style='font-size:20px;font-weight:bold;color:#FFFFFF;'>Universidad de Cundinamarca</span>
        </div>

        <style>
        #boton-enviar {{
            background-color: #014023 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            cursor: pointer !important;
            transition: background-color 0.3s ease !important;
        }}
        #boton-enviar:hover {{
            background-color: #016e33 !important;
        }}
        </style>
    """)

    


    gr.Markdown(
        "<div style='display:flex;align-items:center;'>"
        
        "<h1 style='color:#014023;'> Chatbot Detector de Plantas</h1></div>"
    )

    with gr.Row():
        chatbot = gr.Chatbot(height=500, label="Conversación", type="messages")
        imagen = gr.Image(label="Imagen", type="pil")

    texto = gr.Textbox(label="Mensaje", placeholder="Escribe una pregunta o sube una imagen para analizar.")
    enviar = gr.Button("Enviar",  elem_id="boton-enviar")
    limpiar = gr.Button("Limpiar Chat")
    estado = gr.State({})

    enviar.click(bot_responder, inputs=[texto, chatbot, imagen, estado], outputs=[chatbot, estado])
    limpiar.click(lambda: ([], {}), None, [chatbot, estado])

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
