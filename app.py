#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import render_template, request 
import joblib

@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        purchases = request.form.get("purchases")
        suppcard = request.form.get("suppcard")
        purchases = float(purchases)
        suppcard = float(suppcard)
        print(purchases, suppcard)
        model1 = joblib.load("CCU_DI")
        pred1 = model1.predict([[purchases, suppcard]])
        print(pred1)
        s1 = "The score of credit card upgrade based on decision tree is " + str(pred1)
        model2 = joblib.load("CCU_GB")
        pred2 = model2.predict([[purchases, suppcard]])
        print(pred2)
        s2 = "The score of credit card upgrade based on gradient boost is " + str(pred2)
        model3 = joblib.load("CCU_NN")
        pred3 = model3.predict([[purchases, suppcard]])
        print(pred3)
        s3 = "The score of credit card upgrade based on neural model is " + str(pred3)
        model4 = joblib.load("CCU_Reg")
        pred4 = model4.predict([[purchases, suppcard]])
        print(pred4)
        s4 = "The score of credit card upgrade based on regression model is " + str(pred4)
        model5 = joblib.load("CCU_RF")
        pred5 = model5.predict([[purchases, suppcard]])
        print(pred5)
        s5 = "The score of credit card upgrade based on random forest is " + str(pred5)
        return(render_template("index.html", result1=s1,result2=s2,result3=s3,result4=s4,result5=s5))
    else: 
        return(render_template("index.html", result1="s1",result2="2",result3="2",result4="2",result5="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()
    


# In[ ]:




