import os
import csv
import uuid
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

class DetectorDePlantas:
    """
    Clase principal para la detección de especies vegetales usando YOLO.
    - Detecta objetos en imágenes.
    - Guarda resultados en CSV.
    - Muestra detalles de clases y detecciones.
    """

    def __init__(self, modelo_path: str, confianza_minima: float = 0.50):
        """
        Inicializa el detector cargando el modelo YOLO.
        :param modelo_path: Ruta del archivo .pt del modelo entrenado.
        :param confianza_minima: Nivel mínimo de confianza para aceptar una detección.
        """
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"❌ No se encontró el modelo en: {modelo_path}")

        self.modelo_path = modelo_path
        self.confianza_minima = confianza_minima
        self.modelo = YOLO(modelo_path)
        self.clases = self.modelo.model.names  # Diccionario {id: nombre_clase}

        print(" Modelo cargado correctamente")
        print(f" Número de clases detectables: {len(self.clases)}")
        print(" Clases detectables:")
        for idx, nombre in self.clases.items():
            print(f"  - {idx}: {nombre}")

    def detectar(self, ruta_imagen: str, carpeta_salida: str = "resultados") -> str:
        """
        Detecta especies en una imagen, guarda un CSV con los resultados y los imprime.
        :param ruta_imagen: Ruta de la imagen a procesar.
        :param carpeta_salida: Carpeta donde guardar el CSV.
        :return: Ruta del archivo CSV generado.
        """
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

        # Crear carpeta de resultados si no existe
        os.makedirs(carpeta_salida, exist_ok=True)

        print("\n Iniciando detección...")
        resultados = self.modelo.predict(
            source=ruta_imagen,
            conf=self.confianza_minima,
            save=False,
            verbose=False
        )

        detecciones = []
        conteo_por_clase = {}

        for resultado in resultados:
            for box in resultado.boxes:
                clase_id = int(box.cls[0])
                clase_nombre = self.clases.get(clase_id, "Desconocido")
                confianza = float(box.conf[0])
                bbox = [float(x) for x in box.xyxy[0].tolist()]  # Coordenadas [x1, y1, x2, y2]

                detecciones.append({
                    "clase_id": clase_id,
                    "clase_nombre": clase_nombre,
                    "confianza": round(confianza, 3),
                    "bbox": bbox
                })

                conteo_por_clase[clase_nombre] = conteo_por_clase.get(clase_nombre, 0) + 1

        # Crear nombre único para CSV
        nombre_csv = f"reporte_{uuid.uuid4().hex[:8]}.csv"
        ruta_csv = os.path.join(carpeta_salida, nombre_csv)

        # Guardar resultados en CSV
        self._guardar_csv(ruta_csv, detecciones)

        # Mostrar resumen de resultados
        print("\n RESULTADOS DE LA DETECCIÓN")
        print(f"Imagen procesada: {os.path.basename(ruta_imagen)}")
        print(f" Total de detecciones: {len(detecciones)}")
        print(" Detecciones por clase:")
        for clase, cantidad in conteo_por_clase.items():
            print(f"  - {clase}: {cantidad}")

        print(f"\n Archivo CSV guardado en: {ruta_csv}")
        return ruta_csv

    def _guardar_csv(self, ruta_csv: str, detecciones: list):
        """
        Guarda las detecciones en un archivo CSV.
        :param ruta_csv: Ruta del archivo CSV a generar.
        :param detecciones: Lista de diccionarios con detecciones.
        """
        with open(ruta_csv, mode='w', newline='', encoding='utf-8') as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(["Clase ID", "Clase Nombre", "Confianza", "BBox [x1, y1, x2, y2]"])
            for det in detecciones:
                escritor.writerow([
                    det["clase_id"],
                    det["clase_nombre"],
                    det["confianza"],
                    det["bbox"]
                ])

    def listar_clases(self):
        """
        Muestra todas las clases detectables por el modelo.
        """
        print("\n📚 LISTA DE CLASES DETECTABLES:")
        for idx, nombre in self.clases.items():
            print(f"  - {idx}: {nombre}")



if __name__ == "__main__":
    # Ruta al modelo entrenado
    detector = DetectorDePlantas("runs/detect/especies_colombianas/weights/best.pt")

    # Ruta a la imagen a analizar
    ruta_csv = detector.detectar(
        ruta_imagen=r"C:\Users\julia\Downloads\Red Neuronal YOLO - copia\dataset_especies_colombianas-2\test\images",
        carpeta_salida="reportes"
    )

    # Mostrar ruta del CSV
    print("\n Proceso completado.")
    print(f"CSV generado en: {ruta_csv}")