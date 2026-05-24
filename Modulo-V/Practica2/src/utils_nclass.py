# Sistema
import sys
import json
from pathlib import Path

# Manipulación de datos
import numpy as np
import pandas as pd

# Visualización
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Interactividad en notebook
import ipywidgets as widgets
from IPython.display import display, clear_output

# Configuración pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

# Revisión de versiones
print("Python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("plotly:", plotly.__version__)
print("ipywidgets:", widgets.__version__)

class eda_class:
    """
    Clase para análisis exploratorio visual usando Plotly.

    Permite analizar variables continuas, discretas y relaciones entre variables,
    tomando como base un dataframe principal y un dataframe resumen de campos
    con la columna 'clasificacion'.
    """

    def __init__(
        self,
        df,
        resumen_campos,
        target=None,
        id_col=None,
        output_dir="img"
    ):
        self.df = df.copy()
        self.resumen_campos = resumen_campos.copy()
        self.target = target
        self.id_col = id_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validar_resumen_campos()

    def _validar_resumen_campos(self):
        """
        Valida que el dataframe resumen tenga las columnas mínimas necesarias.
        """

        columnas_requeridas = {"campo", "clasificacion"}

        columnas_faltantes = columnas_requeridas - set(self.resumen_campos.columns)

        if columnas_faltantes:
            raise ValueError(
                f"Faltan columnas en resumen_campos: {columnas_faltantes}"
            )

    def obtener_columnas(self, clasificacion):
        """
        Obtiene columnas según su clasificación: id, target, continua o discreta.
        """

        columnas = (
            self.resumen_campos
            .loc[self.resumen_campos["clasificacion"] == clasificacion, "campo"]
            .tolist()
        )

        columnas = [col for col in columnas if col in self.df.columns]

        return columnas

    def resumen_continuas(self):
        """
        Genera resumen estadístico para variables continuas.
        """

        continuas = self.obtener_columnas("continua")

        if len(continuas) == 0:
            print("No se encontraron variables continuas.")
            return pd.DataFrame()

        resumen = self.df[continuas].describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).T

        return resumen

    def resumen_categoricas(self):
        """
        Genera una tabla de frecuencia y porcentaje para variables discretas.
        """

        discretas = self.obtener_columnas("discreta")

        if len(discretas) == 0:
            print("No se encontraron variables discretas.")
            return pd.DataFrame()

        tablas = []

        for col in discretas:
            tmp = (
                self.df[col]
                .value_counts(dropna=False)
                .reset_index()
            )

            tmp.columns = ["categoria", "frecuencia"]
            tmp["variable"] = col
            tmp["porcentaje"] = tmp["frecuencia"] / len(self.df) * 100
            tmp["porcentaje"] = tmp["porcentaje"].round(2)

            tmp = tmp[["variable", "categoria", "frecuencia", "porcentaje"]]
            tablas.append(tmp)

        return pd.concat(tablas, ignore_index=True)

    def tabla_percentiles(self, col):
        """
        Genera tabla de percentiles para una variable continua.
        """

        self._validar_columna(col)

        percentiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]

        tabla = (
            self.df[col]
            .quantile(percentiles)
            .reset_index()
        )

        tabla.columns = ["percentil", "valor"]
        tabla["percentil"] = (tabla["percentil"] * 100).round(0).astype(int)

        return tabla

    def plot_histograma(self, col, bins=50):
        """
        Genera histograma para una variable continua.
        """

        self._validar_columna(col)

        fig = px.histogram(
            self.df,
            x=col,
            nbins=bins,
            title=f"Histograma de {col}",
            marginal="box"
        )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Frecuencia",
            template="plotly_white"
        )

        return fig

    def plot_histograma_log(self, col, bins=50):
        """
        Genera histograma con transformación log1p para una variable continua.
        """

        self._validar_columna(col)

        df_tmp = self.df[[col]].copy()
        df_tmp[f"log1p_{col}"] = np.log1p(df_tmp[col].clip(lower=0))

        fig = px.histogram(
            df_tmp,
            x=f"log1p_{col}",
            nbins=bins,
            title=f"Histograma log1p de {col}",
            marginal="box"
        )

        fig.update_layout(
            xaxis_title=f"log1p({col})",
            yaxis_title="Frecuencia",
            template="plotly_white"
        )

        return fig

    def plot_histograma_normalizado(self, col, bins=50):
        """
        Genera histograma normalizado en porcentaje para una variable continua.
        """

        self._validar_columna(col)

        fig = px.histogram(
            self.df,
            x=col,
            nbins=bins,
            histnorm="percent",
            title=f"Histograma normalizado de {col}",
            marginal="box"
        )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Porcentaje",
            template="plotly_white"
        )

        return fig

    def plot_boxplot(self, col):
        """
        Genera boxplot para una variable continua.
        """

        self._validar_columna(col)

        fig = px.box(
            self.df,
            y=col,
            points="outliers",
            title=f"Boxplot de {col}"
        )

        fig.update_layout(
            yaxis_title=col,
            template="plotly_white"
        )

        return fig

    def plot_barras_cat_pct(self, col):
        """
        Genera gráfico de barras porcentual para una variable discreta.
        """

        self._validar_columna(col)

        tabla = (
            self.df[col]
            .value_counts(dropna=False, normalize=True)
            .mul(100)
            .reset_index()
        )

        tabla.columns = [col, "porcentaje"]
        tabla["porcentaje"] = tabla["porcentaje"].round(2)
        tabla[col] = tabla[col].astype(str)

        fig = px.bar(
            tabla,
            x=col,
            y="porcentaje",
            text="porcentaje",
            title=f"Distribución porcentual de {col}"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Porcentaje",
            template="plotly_white"
        )

        return fig

    def plot_scatter(self, x_col, y_col):
        """
        Genera gráfico de dispersión entre dos variables continuas.
        """

        self._validar_columna(x_col)
        self._validar_columna(y_col)

        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            title=f"Dispersión: {x_col} vs {y_col}",
            opacity=0.7,
            trendline="ols"
        )

        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_white"
        )

        return fig

    def guardar_figura(self, fig, nombre, formato="html"):
        """
        Guarda una figura de Plotly en la carpeta de salida.

        Por defecto guarda en formato HTML para mantener interactividad.
        """

        nombre_limpio = self._limpiar_nombre_archivo(nombre)

        if formato != "html":
            raise ValueError("Por ahora solo se permite guardar en formato html.")

        output_path = self.output_dir / f"{nombre_limpio}.html"

        fig.write_html(
            output_path,
            include_plotlyjs="cdn",
            full_html=True
        )

        print(f"Gráfico guardado en: {output_path}")

        return output_path

    def _validar_columna(self, col):
        """
        Valida que una columna exista en el dataframe.
        """

        if col not in self.df.columns:
            raise ValueError(f"La columna '{col}' no existe en el dataframe.")

    def _limpiar_nombre_archivo(self, nombre):
        """
        Limpia el nombre del archivo para evitar caracteres problemáticos.
        """

        nombre = str(nombre)
        nombre = nombre.replace(" ", "_")
        nombre = nombre.replace("/", "_")
        nombre = nombre.replace("\\", "_")
        nombre = nombre.replace(":", "_")
        nombre = nombre.replace(".", "_")

        return nombre

class eda_dashboard_class:
    """
    Dashboard interactivo independiente para EDA.

    Usa un objeto eda_class previamente creado.
    No redibuja los widgets; solo oculta o muestra paneles.
    Esto mejora la estabilidad en Jupyter Notebook clásico.
    """

    def __init__(self, eda):
        self.eda = eda
        self.fig_actual = None
        self.nombre_actual = None

        self.continuas = self.eda.obtener_columnas("continua")
        self.discretas = self.eda.obtener_columnas("discreta")

        self.plot_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="560px",
                border="1px solid #ddd",
                padding="8px"
            )
        )

        self.msg_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="45px"
            )
        )

        self._crear_widgets()
        self._crear_paneles()
        self._configurar_eventos()

    def _crear_widgets(self):
        """
        Crea widgets del dashboard.
        """

        self.tipo_analisis = widgets.ToggleButtons(
            options=[
                "Continua",
                "Discreta",
                "Continua vs continua"
            ],
            value="Continua",
            description="Análisis:",
            layout=widgets.Layout(width="330px")
        )

        self.grafico_continua = widgets.Select(
            options=[
                "Histograma",
                "Histograma log",
                "Histograma normalizado",
                "Boxplot",
                "Tabla percentiles"
            ],
            value="Histograma",
            description="Gráfico:",
            rows=5,
            layout=widgets.Layout(width="310px")
        )

        self.grafico_discreta = widgets.Select(
            options=[
                "Barras porcentuales",
                "Tabla frecuencias"
            ],
            value="Barras porcentuales",
            description="Gráfico:",
            rows=2,
            layout=widgets.Layout(width="310px")
        )

        self.variable_continua = widgets.Select(
            options=self.continuas,
            value=self.continuas[0] if len(self.continuas) > 0 else None,
            description="Variable:",
            rows=min(12, max(3, len(self.continuas))),
            layout=widgets.Layout(width="310px", height="230px")
        )

        self.variable_discreta = widgets.Select(
            options=self.discretas,
            value=self.discretas[0] if len(self.discretas) > 0 else None,
            description="Variable:",
            rows=min(10, max(3, len(self.discretas))),
            layout=widgets.Layout(width="310px", height="180px")
        )

        self.variable_x = widgets.Select(
            options=self.continuas,
            value=self.continuas[0] if len(self.continuas) > 0 else None,
            description="X:",
            rows=min(12, max(3, len(self.continuas))),
            layout=widgets.Layout(width="310px", height="220px")
        )

        self.variable_y = widgets.Select(
            options=self.continuas,
            value=self.continuas[1] if len(self.continuas) > 1 else (
                self.continuas[0] if len(self.continuas) > 0 else None
            ),
            description="Y:",
            rows=min(12, max(3, len(self.continuas))),
            layout=widgets.Layout(width="310px", height="220px")
        )

        self.bins = widgets.IntSlider(
            value=50,
            min=10,
            max=100,
            step=5,
            description="Bins:",
            continuous_update=False,
            layout=widgets.Layout(width="310px")
        )

        self.boton_guardar = widgets.Button(
            description="Guardar HTML",
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="310px")
        )

    def _crear_paneles(self):
        """
        Crea paneles independientes para cada tipo de análisis.
        """

        self.panel_continuas = widgets.VBox([
            self.grafico_continua,
            self.variable_continua,
            self.bins
        ])

        self.panel_discretas = widgets.VBox([
            self.grafico_discreta,
            self.variable_discreta
        ])

        self.panel_scatter = widgets.VBox([
            widgets.HTML("<b>Variable continua eje X</b>"),
            self.variable_x,
            widgets.HTML("<b>Variable continua eje Y</b>"),
            self.variable_y
        ])

        self.panel_control = widgets.VBox(
            [
                widgets.HTML("<h3 style='margin-bottom:8px;'>Panel EDA</h3>"),
                self.tipo_analisis,
                widgets.HTML("<hr>"),
                self.panel_continuas,
                self.panel_discretas,
                self.panel_scatter,
                widgets.HTML("<hr>"),
                self.boton_guardar,
                self.msg_output
            ],
            layout=widgets.Layout(
                width="360px",
                border="1px solid #ccc",
                padding="12px",
                margin="0px 14px 0px 0px"
            )
        )

        self._actualizar_visibilidad_paneles()

    def _configurar_eventos(self):
        """
        Configura eventos automáticos.
        """

        self.tipo_analisis.observe(self._evento_cambio_tipo, names="value")

        widgets_a_observar = [
            self.grafico_continua,
            self.grafico_discreta,
            self.variable_continua,
            self.variable_discreta,
            self.variable_x,
            self.variable_y,
            self.bins
        ]

        for widget in widgets_a_observar:
            widget.observe(self._evento_cambio_opcion, names="value")

        self.boton_guardar.on_click(self._guardar_grafico)

    def _evento_cambio_tipo(self, change=None):
        """
        Cambia visibilidad del panel y actualiza el gráfico.
        """

        self._actualizar_visibilidad_paneles()
        self._actualizar_grafico()

    def _evento_cambio_opcion(self, change=None):
        """
        Actualiza gráfico automáticamente al cambiar una opción.
        """

        self._actualizar_grafico()

    def _actualizar_visibilidad_paneles(self):
        """
        Muestra únicamente el panel correspondiente al tipo de análisis.
        """

        tipo = self.tipo_analisis.value

        self.panel_continuas.layout.display = "none"
        self.panel_discretas.layout.display = "none"
        self.panel_scatter.layout.display = "none"

        if tipo == "Continua":
            self.panel_continuas.layout.display = "flex"

        elif tipo == "Discreta":
            self.panel_discretas.layout.display = "flex"

        elif tipo == "Continua vs continua":
            self.panel_scatter.layout.display = "flex"

    def _actualizar_grafico(self):
        """
        Actualiza el gráfico o tabla según la selección actual.
        """

        tipo = self.tipo_analisis.value

        self.plot_output.clear_output(wait=True)

        with self.plot_output:
            try:
                if tipo == "Continua":
                    self._generar_continua()

                elif tipo == "Discreta":
                    self._generar_discreta()

                elif tipo == "Continua vs continua":
                    self._generar_scatter()

            except Exception as error:
                self.fig_actual = None
                self.nombre_actual = None
                print(f"Error al generar visualización: {error}")

    def _generar_continua(self):
        """
        Genera visualización para variable continua.
        """

        col = self.variable_continua.value
        grafico = self.grafico_continua.value
        bins = self.bins.value

        if grafico == "Histograma":
            fig = self.eda.plot_histograma(col, bins=bins)
            nombre = f"hist_{col}"

        elif grafico == "Histograma log":
            fig = self.eda.plot_histograma_log(col, bins=bins)
            nombre = f"hist_log_{col}"

        elif grafico == "Histograma normalizado":
            fig = self.eda.plot_histograma_normalizado(col, bins=bins)
            nombre = f"hist_norm_{col}"

        elif grafico == "Boxplot":
            fig = self.eda.plot_boxplot(col)
            nombre = f"box_{col}"

        elif grafico == "Tabla percentiles":
            tabla = self.eda.tabla_percentiles(col)

            self.fig_actual = None
            self.nombre_actual = None

            display(tabla)
            self._mostrar_mensaje(f"Tabla generada: percentiles de {col}")
            return

        fig.update_layout(
            width=950,
            height=520,
            margin=dict(l=40, r=40, t=70, b=40)
        )

        self.fig_actual = fig
        self.nombre_actual = nombre

        display(fig)
        self._mostrar_mensaje(f"Visualización generada: {grafico} - {col}")

    def _generar_discreta(self):
        """
        Genera visualización para variable discreta.
        """

        col = self.variable_discreta.value
        grafico = self.grafico_discreta.value

        if grafico == "Barras porcentuales":
            fig = self.eda.plot_barras_cat_pct(col)
            nombre = f"bar_pct_{col}"

            fig.update_layout(
                width=950,
                height=520,
                margin=dict(l=40, r=40, t=70, b=40)
            )

            self.fig_actual = fig
            self.nombre_actual = nombre

            display(fig)
            self._mostrar_mensaje(f"Visualización generada: {grafico} - {col}")

        elif grafico == "Tabla frecuencias":
            tabla = (
                self.eda.resumen_categoricas()
                .query("variable == @col")
                .reset_index(drop=True)
            )

            self.fig_actual = None
            self.nombre_actual = None

            display(tabla)
            self._mostrar_mensaje(f"Tabla generada: frecuencias de {col}")

    def _generar_scatter(self):
        """
        Genera gráfico de dispersión continua vs continua.
        """

        x_col = self.variable_x.value
        y_col = self.variable_y.value

        fig = self.eda.plot_scatter(x_col, y_col)
        nombre = f"scatter_{x_col}_vs_{y_col}"

        fig.update_layout(
            width=950,
            height=520,
            margin=dict(l=40, r=40, t=70, b=40)
        )

        self.fig_actual = fig
        self.nombre_actual = nombre

        display(fig)
        self._mostrar_mensaje(f"Visualización generada: {x_col} vs {y_col}")

    def _guardar_grafico(self, button=None):
        """
        Guarda el gráfico actual en HTML.
        """

        if self.fig_actual is None or self.nombre_actual is None:
            self._mostrar_mensaje(
                "No hay gráfico para guardar. Las tablas no se guardan con este botón."
            )
            return

        output_path = self.eda.guardar_figura(
            fig=self.fig_actual,
            nombre=self.nombre_actual,
            formato="html"
        )

        self._mostrar_mensaje(f"Gráfico guardado en: {output_path}")

    def _mostrar_mensaje(self, mensaje):
        """
        Muestra mensajes dentro del panel.
        """

        self.msg_output.clear_output(wait=True)

        with self.msg_output:
            print(mensaje)

    def mostrar(self):
        """
        Muestra el dashboard completo.
        """

        contenedor = widgets.HBox(
            [
                self.panel_control,
                self.plot_output
            ],
            layout=widgets.Layout(
                width="100%",
                align_items="flex-start"
            )
        )

        display(contenedor)

        self._actualizar_grafico()