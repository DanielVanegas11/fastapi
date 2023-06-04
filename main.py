from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
    
class PredictionOut(BaseModel):
    answer: str

class TextIn(BaseModel):
  J1_2ndWon: int
  J1_ace: int
  J1_bpFaced: int
  J1_df: int
  J1_age: int
  J1_ht: int
  J1_rank: int
  J2_2ndWon: int
  J2_ace: int
  J2_bpFaced: int
  J2_df: int
  J2_age: int
  J2_ht: int
  J2_rank: int
  J1_hand_R: int
  surface_Clay: int
  surface_Grass: int
  surface_Hard: int
  tourney_level_G: int
  tourney_level_M: int
  J2_hand_R: int




@app.post("/prediccion")

async def Prediccion_resultado(payload: TextIn): 
    """"
    Esta aplicación está diseñada para predecir el resultado de un partido de tenis de campo pasando como argumentos las estadísticas del partido y de los jugadores para así mediante un modelo determinar si el jugador ganaría "G" o perdería ("P") el partido. \n
    \n
    A continuacuón se presenta una descripción de las variables que debe recibir el modelo para realizar las predicciones. \n
    \n
    Todas las variables que contengan en la descripción "J1" corresponden a estadísticas asociadas al jugador 1, que es sobre el cual el modelo predice si gana o pierde, mientras que las descripciones que contengan "J2" corresponderán a las estadísticas del jugador 2.
    

    J1_2ndWon: Puntos ganados con el segundo servicio. \n
    J1_ace: Saques directos. \n
    J1_bpFaced: Puntos de quiebre enfrentados.\n
    J1_df: Dobles faltas.\n
    J1_age: Edad.\n
    J1_ht: Altura.\n
    J1_rank: Posición en el ranking ATP.\n
    J2_2ndWon: Puntos ganados con el segundo servicio. \n
    J2_ace: Saques directos.\n
    J2_bpFaced: Puntos de quiebre enfrentados.\n
    J2_df: Dobles faltas.\n
    J2_age: Edad.\n
    J2_ht: Altura.\n
    J2_rank: Posición en el ranking ATP.\n
    J1_hand_R: Mano hábil del jugador (Diestro=1, Zurdo=0).\n
    surface_Clay: Indicar 1 si el partido se juega en polvo de ladrillo, 0 en cualquier otro caso.\n
    surface_Grass: Indicar 1 si el partido se juega en cesped, 0 en cualquier otro caso.\n
    surface_Hard: Indicar 1 si el partido se juega en cemento, 0 en cualquier otro caso.\n
    tourney_level_G: Indicar 1 si el torneo es Grand Slam, 0 en cualquier otro caso\n
    tourney_level_M: Indicar 1 si el torneo es Masters 1000, 0 en cualquier otro caso\n
    J2_hand_R: Mano hábil del jugador (Diestro=1, Zurdo=0).\n

    """
    modelo = joblib.load('resources/model.joblib')
    
    escalador = joblib.load('resources/scaler.joblib')

    datos=pd.DataFrame(dict(payload),index=[0])

    datos_escalados=escalador.transform(datos)

    Resultado=modelo.predict(datos_escalados)


    return {"Resultado":Resultado.item(0)}
