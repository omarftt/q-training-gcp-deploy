from pydantic import BaseModel

class LoanPredictionRequest(BaseModel):
    Age: float
    Annual_Income: float
    Credit_Score: float
    Loan_Amount: float
    Number_of_Open_Accounts: float