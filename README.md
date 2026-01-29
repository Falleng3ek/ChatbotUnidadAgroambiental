# 🤖🌿 ChatbotUnidadAgroambiental

## 📌 Descripción del Proyecto
ChatbotUnidadAgroambiental es un sistema inteligente desarrollado para la identificación de especies vegetales en la Unidad Agroambiental La Esperanza. El proyecto integra visión por computadora mediante un modelo YOLOv8 entrenado con especies locales y un módulo tipo chatbot que permite interactuar con el usuario para brindar información sobre las plantas detectadas.

Este sistema forma parte de una iniciativa académica orientada a la aplicación de tecnologías de Inteligencia Artificial en el monitoreo agroambiental.

---

## 🎯 Objetivo
Desarrollar una herramienta tecnológica que permita:
- Detectar especies vegetales mediante imágenes
- Proporcionar información relevante de cada especie
- Facilitar procesos educativos y de investigación en entornos agroambientales

---

## 🚀 Estado del Proyecto
Prototipo funcional en desarrollo académico  
Proyecto de grado – Ingeniería de Sistemas y Computación  

---

## 🧠 Funcionalidades Principales
- Detección de especies vegetales con YOLOv8
- Interacción tipo chatbot para consultas sobre especies
- Generación de reportes de detección
- Procesamiento de imágenes locales
- Base de datos adaptable para nuevas especies

---

## 🛠 Tecnologías Utilizadas
- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- Pillow (PIL)
- JSON para base de conocimientos
- Modelos de Deep Learning CNN
- Raspberry Pi (opcional para implementación en campo)

---

## 📂 Estructura del Proyecto

📦 ChatbotUnidadAgroambiental
├── chatbot_main.py        Script principal del chatbot
├── detector.py            Lógica de detección con YOLO
├── questions.json         Preguntas frecuentes
├── answers.json           Respuestas del chatbot
├── yolov8_model.pt        Pesos del modelo entrenado
├── logo.jpeg              Imagen representativa
├── reportes/              Resultados de detección
├── runs/                  Resultados de entrenamiento/pruebas
└── tmp/                   Archivos temporales

---

## ⚙️ Instalación

1. Clonar el repositorio
git clone https://github.com/Falleng3ek/ChatbotUnidadAgroambiental.git

2. Ingresar al directorio del proyecto
cd ChatbotUnidadAgroambiental

3. Crear entorno virtual (opcional)
python -m venv venv

4. Activar entorno virtual
Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

5. Instalar dependencias
pip install -r requirements.txt

6. Recordar Cambiar las rutas a las correspondientes a su Ordenador y entorno
---

## ▶️ Uso

1. Colocar imágenes en la carpeta designada
2. Ejecutar el sistema
python chatbot_main.py
3. Seguir instrucciones en consola para interactuar con el chatbot
4. Consultar resultados en la carpeta reportes/

---

## 📊 Resultados Esperados
- Identificación automática de especies vegetales
- Precisión en detección basada en entrenamiento del modelo
- Interacción amigable con el usuario

---

## 🔮 Mejoras Futuras
- Integración con aplicación móvil
- Conexión a base de datos en la nube
- Mejora de precisión del modelo
- Implementación en tiempo real con cámaras IoT
- Panel web para visualización de datos

---

## 🤝 Contribuciones
Las contribuciones son bienvenidas:
1. Fork del repositorio
2. Crear nueva rama
3. Realizar cambios
4. Enviar Pull Request

---

## 👨‍💻 Autor
Proyecto desarrollado por:
Falleng3ek  
Estudiante de Ingeniería de Sistemas y Computación  

---

## 📜 Licencia
Este proyecto no cuenta actualmente con una licencia de uso.
Todos los derechos están reservados al autor.

El código se publica únicamente con fines académicos y de consulta.
No se autoriza su uso comercial, modificación o redistribución sin permiso previo.


---

## 📬 Contacto
GitHub: https://github.com/Falleng3ek

---

## ⭐ Agradecimientos
A la Unidad Agroambiental La Esperanza por facilitar los espacios y datos para el desarrollo del proyecto.

---

Aplicando Inteligencia Artificial para fortalecer la identificación y conservación de especies vegetales.
