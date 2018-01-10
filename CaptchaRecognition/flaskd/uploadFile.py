# coding:utf-8
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
from resu import *
from PIL import Image
from multiprocessing import Pool
import os,time

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def upload(output='Output Result'):
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, './static/uploads',secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        img = Image.open(upload_path)
        img = img.resize((40,40))
        img = np.array(img.convert('1'))
        img.shape = 1,1600

        pool = Pool()
        out = pool.map(cnnmain, [img])
        output = out[0]

        return render_template('index.html', output=output)
    return render_template('index.html', output="Output_Result")

if __name__ == '__main__':
    app.run(debug=True)
