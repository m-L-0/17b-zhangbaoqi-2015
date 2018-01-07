# coding:utf-8
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
from resu import *
from PIL import Image
import os

app = Flask(__name__)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, './static/uploads',secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        img = Image.open(upload_path)
        img = img.resize((40,40))
        img = np.array(img.convert('1'))
        img.shape = 1,1600
        output = cnnmain(img)

        return redirect(url_for('result', output=output))
    return render_template('upload.html')
@app.route('/result/<output>')
def result(output):
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
