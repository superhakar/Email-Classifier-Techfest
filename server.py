import uvicorn
from fastapi import FastAPI, File, UploadFile
import model
import fun
import os, zipfile
import shutil
from starlette.responses import FileResponse

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        with zipfile.ZipFile(file.filename) as zip_file:
            zip_file.extractall('Dataset/')
    os.remove(file.filename)
    data,code = fun.convert()
    model.preprocess(data,'train')
    f = open("Code/code.txt",'w+')
    for i in range(len(code)):
        f.write("%s\r\n" % code[i])
    f.close()
    return 

@app.post("/train")
async def train():
    oob,cv = model.train()
    return "Success"

@app.post("/test")
async def test(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        with zipfile.ZipFile(file.filename) as zip_file:
            zip_file.extractall('Testset/')
    os.remove(file.filename)
    data = fun.converttest()
    model.preprocess(data,test)
    model.test()
    return download("Result/result.csv")

def download(file_path):
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return None

if __name__ == "__main__":
    uvicorn.run(app,host = '127.0.0.1',port=8080)