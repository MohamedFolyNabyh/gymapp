from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle


# تحميل النموذج المحفوظ
with open(r'C:\Users\DELL\Downloads\multioutput_model.pkl', 'rb') as f:
    model = pickle.load(f)


# # إنشاء تطبيق FastAPI
app = FastAPI()

# نموذج إدخال البيانات
class FitnessData(BaseModel):
    sex: str
    age: int
    height: float
    weight: float
    diabetes: str
    hypertension: str

# حساب BMI
def calculate_bmi(weight, height):
    return round(weight / (height ** 2), 2)

# تحديد الحقول بناءً على BMI
def determine_fitness_plan(bmi):
    if bmi < 18.5:
        return "Underweight", "Weight Gain", "Muscular Fitness"
    elif 18.5 <= bmi < 24.9:
        return "Normal", "Weight Gain", "Muscular Fitness"
    elif 25 <= bmi < 29.9:
        return "Overweight", "Weight Loss", "Cardio Fitness"
    else:
        return "Obese", "Weight Loss", "Cardio Fitness"

@app.post("/predict/")
async def predict(data: FitnessData):
    # حساب BMI
    bmi = calculate_bmi(data.weight, data.height)
    level, fitness_goal, fitness_type = determine_fitness_plan(bmi)

    # إعداد بيانات الإدخال للنموذج
    input_data = {
        "Sex": [1 if data.sex.lower() == 'male' else 0],
        "Age": [data.age],
        "Height": [data.height],
        "Weight": [data.weight],
        "BMI": [bmi],
        "Diabetes": [1 if data.diabetes.lower() == "yes" else 0],
        "Hypertension": [1 if data.hypertension.lower() == "yes" else 0],
        "Fitness Goal": [fitness_goal],
        "Level": [level],
        "Fitness Type": [fitness_type]
    }
    df = pd.DataFrame(input_data)

    # تحويل النصوص إلى أرقام
    df["Level"] = df["Level"].astype("category").cat.codes
    df["Fitness Goal"] = df["Fitness Goal"].astype("category").cat.codes
    df["Fitness Type"] = df["Fitness Type"].astype("category").cat.codes

    try:
        # إجراء التنبؤ
        predictions = model.predict(df)

        
        diet_prediction=int(predictions[0][0])
        exercise_prediction=int(predictions[0][1])

        # إرجاع القيم في الاستجابة
        return {
            "BMI": bmi,
            "Predicted Level": level,
            "Predicted Goal": fitness_goal,
            "Predicted Type": fitness_type,
            "Predicted Diet": diet_prediction,
            "Predicted Exercise": exercise_prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
