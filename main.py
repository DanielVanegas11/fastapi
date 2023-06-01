from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI()

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


  


@app.get("/") #Get es para consultar algo. Post es para recibir algo
async def root():
    return {"message": "Hello World"}


@app.post("/prediccion")

async def prueba(payload: TextIn): 
    """"
    Esta función está diseñada para predecir blah blah y pongo toda la carreta
    """
    modelo = joblib.load('resources/model.joblib')
    
    escalador = joblib.load('resources/scaler.joblib')

    datos=pd.DataFrame(dict(payload),index=[0])

    datos_escalados=escalador.transform(datos)

    Resultado=modelo.predict(datos_escalados)


    return {"Resultado":Resultado.item(0)}