from fastapi import APIRouter, HTTPException
from app.services.model_service import ModelService

router = APIRouter()


@router.post("/train-model/")
def train():
    try:
        ModelService.treinar_modelo()
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/use-model/")
def use(text: str):
    try:
        result = ModelService.usar_modelo(text)
        return {"message": "Model used successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
