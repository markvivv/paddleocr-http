import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.config import dictConfig

import cv2
import numpy as np
from flask import Flask, request
from paddleocr import PaddleOCR

# 配置flask的日志
dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["wsgi"]},
    }
)

app = Flask(__name__)
# 解决中文输出为unicode
app.config["JSON_AS_ASCII"] = False
# 解决浏览器中中文乱码
app.config["JSONIFY_MIMETYPE"] = "application/json;charset=UTF-8"
# 将flask的日志对象设置为常用的logger别名
logger = app.logger


def upload_image(bytes, filename=None, mime_type=None):

    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(
        use_angle_cls=True, lang="ch"
    )  # need to run only once to download and load model into memory

    # 转换图片二进制为ocr接口需要的ndarray数据结构
    np_arr = np.frombuffer(bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 执行图片识别
    result = ocr.ocr(img, cls=True)
    logger.debug(result)
    ocr_result = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            ocr_result.append(line[1])

    return ocr_result


@app.route("/hello")
def hello():
    return f"Hello!"


"""执行图像上的文字识别，返回识别的文字以及对应文字识别的置信值，支持自动旋转图片"""


@app.route("/v1/img/ocr", methods=["GET", "POST"])
def api_v1_upload_img():
    req_start = time.time()
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file_name = file.filename
            # 获取图片的二进制
            res = upload_image(bytes=file.read(), filename=file_name)

            req_elapsed = time.time() - req_start
            logger.info(
                "Image %s OCR Took %.2f seconds. result: %s",
                file.filename,
                req_elapsed,
                res,
            )
            return res
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """
