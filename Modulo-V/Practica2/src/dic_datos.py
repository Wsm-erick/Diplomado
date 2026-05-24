from pathlib import Path
import json


DICCIONARIO_DATOS = {
    "ACCOUNT_ID": "Identificador único de cada cliente.",
    "LIMIT_BAL": "Monto de la línea de crédito otorgada al cliente.",
    "SEX": "Género del cliente: 1 = masculino, 2 = femenino.",
    "EDUCATION": (
        "Nivel educativo: 1 = preparatoria, 2 = universidad, "
        "3 = secundaria, 4 = otros, 5 = desconocido, "
        "6 = desconocido, 0 = nulo."
    ),
    "MARRIAGE": "Estado marital: 1 = casado, 2 = soltero, 3 = otros, 0 = nulo.",
    "AGE": "Edad del cliente en años.",

    "PAY_0": "Estado de pago en septiembre de 2005.",
    "PAY_2": "Estado de pago en agosto de 2005.",
    "PAY_3": "Estado de pago en julio de 2005.",
    "PAY_4": "Estado de pago en junio de 2005.",
    "PAY_5": "Estado de pago en mayo de 2005.",
    "PAY_6": "Estado de pago en abril de 2005.",

    "BILL_AMT1": "Monto por pagar en septiembre de 2005.",
    "BILL_AMT2": "Monto por pagar en agosto de 2005.",
    "BILL_AMT3": "Monto por pagar en julio de 2005.",
    "BILL_AMT4": "Monto por pagar en junio de 2005.",
    "BILL_AMT5": "Monto por pagar en mayo de 2005.",
    "BILL_AMT6": "Monto por pagar en abril de 2005.",

    "PAY_AMT1": "Cantidad pagada en septiembre de 2005. Variable objetivo para regresión.",
    "PAY_AMT2": "Cantidad pagada en agosto de 2005.",
    "PAY_AMT3": "Cantidad pagada en julio de 2005.",
    "PAY_AMT4": "Cantidad pagada en junio de 2005.",
    "PAY_AMT5": "Cantidad pagada en mayo de 2005.",
    "PAY_AMT6": "Cantidad pagada en abril de 2005.",

    "default.payment.next.month": (
        "Incumplimiento de pago del siguiente mes: "
        "1 = sí incumple, 0 = no incumple."
    )
}


def guardar_diccionario_json(diccionario, output_path):
    """
    Guarda un diccionario de Python en formato JSON.

    Parameters
    ----------
    diccionario : dict
        Diccionario con la descripción de variables.

    output_path : str or pathlib.Path
        Ruta donde se guardará el archivo JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            diccionario,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Diccionario guardado en: {output_path}")


if __name__ == "__main__":

    output_path = Path("config") / "diccionario_datos.json"

    guardar_diccionario_json(
        diccionario=DICCIONARIO_DATOS,
        output_path=output_path
    )