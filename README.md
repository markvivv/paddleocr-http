# 安装PaddlePaddle及使用PP-OCRv3 模型提取身份证信息

## 一、前言

### 1）目标

- 本指南的首要目的是提供一种快速使用百度开源深度学习平台（飞浆平台）的方法，飞浆平台提供了很多已经完成训练的AI模型，让了解 Python、Docker、Linux 基础知识的开发人员能够在**一至几个工作日**内完成飞浆AI模型在项目的使用搭建；
  - 飞浆平台支持的硬件环境：
    - Nvida显卡系列：CUDA 10.2、CUDA 11.2、CUDA 11.6、CUDA 11.7
    - AMD显卡系列：ROCm 4.0
    - CPU

  - 飞浆平台模型开发套件（基于开发套件进行训练会产生不同的模型）
    - 文字识别 - PaddleOCR，身份证、车票、增值税发票、车牌、液晶屏读数、印章等提取文字
    - 视频理解 - PaddleVideo，工业环境中异常行为检测，体育运动中的动作剪辑，互联网场景中的视频质量评估
    - 目标检测 - PaddleDetection，车流统计、车辆违章检测、闯入、表面质量检测
    - 图像分割 - PaddleSeg，道路积水识别、区域变化检测
    - 语音识别 - PaddleSpeech，语音翻译、语音合成、标点恢复
    - 语义理解 - ERNIE，自动问答、情感分析、基于语义的相似度推荐
    - 图神经网络 - PGL，推荐系统、知识图谱、风控、流量预测
    - 时空大数据计算工具 - PaddleSpatial，在城市空间区域画像、智能交管、道路规划场景中提供算法支持

- 当前指南以 PaddlePaddle CPU Docker 镜像为基础，安装 PaddleOCR 模型开发套件，并使用配套的 [PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/PP-OCRv3_introduction.md) 文字识别模型；
- 使用 Python Flask 轻量化 Web 框架完成了 PaddleOCR SDK 能力转换成 HTTP API 服务（非常简单，代码量很小，不足100行），其他模型的 SDK 能力可以参照本实例进行开发；
- 按照生产环境发布的要求，使用 uWSGI 运行 Flask，并构造为一个 Docker 镜像，方便进行发布；
- 镜像构造完成后能够支持离线环境运行。

### 2）未解决的问题

- PaddleOCR 以 PaddleOcloud 的身份在 Dockerhub 发布了 [PaddleOCR](https://hub.docker.com/r/paddlecloud/paddleocr) 镜像，实际测试下来，已知的问题为：默认镜像没有提供paddleocr命令，flask 版本比 paddlepaddle/paddle:2.4.1 低，简单尝试之后暂时放弃，回到了使用 paddlepaddle/paddle:2.4.1作为上游镜像构造本镜像；
- paddleocr 作为模块安装完成后，实际使用仍然需要下载模型库，由于未找到正确放置模块库的方法，暂时在 Dockerfile 中通过执行一次 paddleocr 测试识别来完成模型库的自动下载；
- paddleocr_http.py 做的比较简单，flask 的日志配置没有从程序中独立出来，paddleocr只做了图片接口，没有做pdf文件接口，这些都需要在项目中根据情况去完善。

## 二、Docker 安装 PaddlePaddle

[PaddlePaddle](https://github.com/PaddlePaddle)是飞浆平台的基础运行环境，使用飞浆平台提供的已经完成训练的模型时，需要依赖此基础运行环境。

进入[飞浆快速安装网页](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/macos-docker.html)，按照自己情况选择对应的飞浆版本、操作系统、计算平台，安装方式建议选择 Docker。以下以我自己Intel 苹果笔记本电脑Docker方式安装为例：

创建一个挂载目录，用于容器和宿主机交换文件数据：

```shell
mkdir paddle
cd paddle
```

拉取镜像并进入Docker容器

```shell
docker run --name paddle -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.4.1 /bin/bash
```

## 三、在 PaddlePaddle Docker 环境中安装 PP-OCRv3 模型并执行测试

将测试的身份证照片放入 刚刚创建的paddle 目录，并在容器内进入 /paddle 目录。然后安装 [PP-OCRv3 模型](https://www.paddlepaddle.org.cn/modelsDetail?modelId=17)，参照官方的快速体验方法：

```shell
cd /paddle
python3 -m pip install paddleocr
```

命令输出：

```shell
Collecting paddleocr
  Downloading paddleocr-2.6.1.2-py3-none-any.whl (440 kB)
     |████████████████████████████████| 440 kB 688 kB/s 
Collecting opencv-python
  Downloading opencv_python-4.7.0.68-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (61.8 MB)
     |████████████████████████████████| 61.8 MB 4.6 MB/s 
Collecting scikit-image
  Downloading scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
     |████████████████████████████████| 13.5 MB 4.4 MB/s 
Collecting attrdict
  Downloading attrdict-2.0.1-py2.py3-none-any.whl (9.9 kB)
Collecting lmdb
  Downloading lmdb-1.4.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (299 kB)
     |████████████████████████████████| 299 kB 373 kB/s 
     
     ……
     
     Successfully installed Babel-2.11.0 Flask-Babel-3.0.1 PyMuPDF-1.20.2 PyWavelets-1.3.0 Werkzeug-2.2.2 aiofiles-22.1.0 aiohttp-3.8.3 aiosignal-1.3.1 altair-4.2.2 anyio-3.6.2 async-timeout-4.0.2 asynctest-0.13.0 attrdict-2.0.1 bce-python-sdk-0.8.74 beautifulsoup4-4.11.1 brotli-1.0.9 cachetools-5.3.0 click-8.1.3 cssselect-1.2.0 cssutils-2.6.0 cycler-0.11.0 cython-0.29.33 dill-0.3.6 et-xmlfile-1.1.0 fastapi-0.89.1 ffmpy-0.3.0 fire-0.5.0 flask-2.2.2 fonttools-4.38.0 frozenlist-1.3.3 fsspec-2023.1.0 future-0.18.3 gevent-22.10.2 geventhttpclient-2.0.2 gradio-3.16.2 greenlet-2.0.2 grpcio-1.42.0 h11-0.14.0 httpcore-0.16.3 httpx-0.23.3 imageio-2.25.0 imgaug-0.4.0 importlib-resources-5.10.2 itsdangerous-2.1.2 jinja2-3.1.2 jsonschema-4.17.3 kiwisolver-1.4.4 linkify-it-py-1.0.3 lmdb-1.4.0 lxml-4.9.2 markdown-it-py-2.1.0 markupsafe-2.1.2 matplotlib-3.5.3 mdit-py-plugins-0.3.3 mdurl-0.1.2 mpmath-1.2.1 multidict-6.0.4 multiprocess-0.70.14 networkx-2.6.3 onnx-1.13.0 opencv-contrib-python-4.7.0.68 opencv-python-4.7.0.68 openpyxl-3.0.10 orjson-3.8.5 paddleocr-2.6.1.2 pandas-1.3.5 pdf2docx-0.5.6 pkgutil-resolve-name-1.3.10 premailer-3.10.0 psutil-5.9.4 pyclipper-1.3.0.post4 pycryptodome-3.17 pydantic-1.10.4 pydub-0.25.1 pyrsistent-0.19.3 python-docx-0.8.11 python-multipart-0.0.5 python-rapidjson-1.9 pytz-2022.7.1 rapidfuzz-2.13.7 rarfile-4.0 rfc3986-1.5.0 scikit-image-0.19.3 scipy-1.7.3 shapely-2.0.0 sniffio-1.3.0 soupsieve-2.3.2.post1 starlette-0.22.0 sympy-1.10.1 termcolor-2.2.0 tifffile-2021.11.2 toolz-0.12.0 tqdm-4.64.1 tritonclient-2.29.0 uc-micro-py-1.0.1 uvicorn-0.20.0 visualdl-2.5.0 websockets-10.4 x2paddle-1.4.0 yarl-1.8.2 zope.event-4.6 zope.interface-5.5.2
```

执行身份证图片测试，第一次执行会下载一些必要的依赖，后面再执行就不需要下载了。从测试的情况看，不论身份证图片大小，单次身份证图片识别需要1~2秒钟。

```shell
paddleocr --image_dir ./small.jpg --use_angle_cls true --use_gpu false --lang=ch
```

命令输出：

```shell
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
download https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar to /root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.tar
100%|█████████████████████████████████████████████████████████████████████████████| 3.83M/3.83M [00:01<00:00, 3.43MiB/s]
download https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar to /root/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.tar
100%|█████████████████████████████████████████████████████████████████████████████| 11.9M/11.9M [00:07<00:00, 1.52MiB/s]
download https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar to /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar
100%|█████████████████████████████████████████████████████████████████████████████| 2.19M/2.19M [00:01<00:00, 1.84MiB/s]
[2023/01/30 04:54:30] ppocr DEBUG: Namespace(alpha=1.0, benchmark=False, beta=1.0, cls_batch_num=6, cls_image_shape='3, 48, 192', cls_model_dir='/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer', cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det=True, det_algorithm='DB', det_box_type='quad', det_db_box_thresh=0.6, det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, det_limit_side_len=960, det_limit_type='max', det_model_dir='/root/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer', det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, det_pse_thresh=0, det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results', drop_score=0.5, e2e_algorithm='PGNet', e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast', e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5, gpu_mem=500, help='==SUPPRESS==', image_dir='./small.jpg', image_orientation=False, ir_optim=True, kie_algorithm='LayoutXLM', label_list=['0', '180'], lang='ch', layout=True, layout_dict_path=None, layout_model_dir=None, layout_nms_threshold=0.5, layout_score_threshold=0.5, max_batch_size=10, max_text_length=25, merge_no_span_structure=True, min_subgraph_size=15, mode='structure', ocr=True, ocr_order_method=None, ocr_version='PP-OCRv3', output='./output', page_num=0, precision='fp32', process_id=0, re_model_dir=None, rec=True, rec_algorithm='SVTR_LCNet', rec_batch_num=6, rec_char_dict_path='/usr/local/lib/python3.7/dist-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_model_dir='/root/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer', recovery=False, save_crop_res=False, save_log_path='./log_output/', scales=[8, 16, 32], ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ser_model_dir=None, show_log=True, sr_batch_num=1, sr_image_shape='3, 32, 128', sr_model_dir=None, structure_version='PP-StructureV2', table=True, table_algorithm='TableAttn', table_char_dict_path=None, table_max_len=488, table_model_dir=None, total_process_num=1, type='ocr', use_angle_cls=True, use_dilation=False, use_gpu=False, use_mp=False, use_npu=False, use_onnx=False, use_pdf2docx_api=False, use_pdserving=False, use_space_char=True, use_tensorrt=False, use_visual_backbone=True, use_xpu=False, vis_font_path='./doc/fonts/simfang.ttf', warmup=False)
[2023/01/30 04:54:38] ppocr INFO: **********./small.jpg**********
[2023/01/30 04:54:45] ppocr DEBUG: dt_boxes num : 12, elapse : 6.396355390548706
[2023/01/30 04:54:45] ppocr DEBUG: cls num  : 12, elapse : 0.3474392890930176
[2023/01/30 04:54:50] ppocr DEBUG: rec_res num  : 12, elapse : 4.751603364944458
[2023/01/30 04:54:50] ppocr INFO: [[[259.0, 112.0], [391.0, 112.0], [391.0, 153.0], [259.0, 153.0]], ('***', 0.9297130703926086)]
[2023/01/30 04:54:50] ppocr INFO: [[[142.0, 125.0], [250.0, 121.0], [251.0, 153.0], [143.0, 157.0]], ('姓名', 0.9741479158401489)]
[2023/01/30 04:54:50] ppocr INFO: [[[182.0, 194.0], [488.0, 194.0], [488.0, 229.0], [182.0, 229.0]], ('别男民族汉', 0.991115391254425)]
[2023/01/30 04:54:50] ppocr INFO: [[[162.0, 272.0], [222.0, 272.0], [222.0, 303.0], [162.0, 303.0]], ('生', 0.9976730942726135)]
[2023/01/30 04:54:50] ppocr INFO: [[[248.0, 272.0], [573.0, 269.0], [573.0, 300.0], [248.0, 304.0]], ('19**年*月3日', 0.9517846703529358)]
[2023/01/30 04:54:50] ppocr INFO: [[[140.0, 351.0], [225.0, 351.0], [225.0, 384.0], [140.0, 384.0]], ('住址', 0.9953198432922363)]
[2023/01/30 04:54:50] ppocr INFO: [[[256.0, 354.0], [641.0, 351.0], [642.0, 382.0], [256.0, 384.0]], ('武汉市**区*****', 0.9569936394691467)]
[2023/01/30 04:54:50] ppocr INFO: [[[256.0, 405.0], [610.0, 401.0], [610.0, 435.0], [256.0, 439.0]], ('******-*-****', 0.9409841299057007)]
[2023/01/30 04:54:50] ppocr INFO: [[[137.0, 557.0], [362.0, 555.0], [362.0, 586.0], [138.0, 589.0]], ('公民身份号码', 0.9767408967018127)]
[2023/01/30 04:54:50] ppocr INFO: [[[406.0, 553.0], [949.0, 550.0], [949.0, 584.0], [406.0, 586.0]], ('4***************', 0.9046091437339783)]
```

## 四、以 HTTP 标准接口提供 OCR 识别能力

在 Docker 挂载的 paddle 目录下编写 Python Flask Web 应用，paddleocr_http.py 文件内容如下：

```python
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
        "root": {"level": "DEBUG", "handlers": ["wsgi"]},
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

```

##五、在 Docker 中执行 Flask 开发模式调试

```shell
flask --app paddleocr_http --debug run --host 0.0.0.0 --port 8080
```

命令输出：

```shell
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
 * Serving Flask app 'paddleocr_http'
 * Debug mode: on
[2023-01-31 01:08:54,162] INFO in _internal: WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://172.17.0.2:8080
[2023-01-31 01:08:54,163] INFO in _internal: Press CTRL+C to quit
[2023-01-31 01:08:54,164] INFO in _internal:  * Restarting with stat
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
[2023-01-31 01:08:56,326] WARNING in _internal:  * Debugger is active!
[2023-01-31 01:08:56,341] INFO in _internal:  * Debugger PIN: 108-237-100
```

> 再次以命令行进入前述 Docker 容器的方法：
>
> 1. 确保已经初始化的 Docker 容器是启动状态，若未启动，执行 `docker start paddle_docker` 启动 Docker 容器；
> 2. 使用 `docker ps` 确定 paddle_docker 容器的 ID；
> 3. 执行 `docker attach [CONTAINERID]` 进入容器的命令行；
> 4. 执行 `cd /paddle` 目录启动 Flask 调试模式。

curl测试

```shell
curl -X POST -F file=@./large.jpg http://localhost:8080/v1/img/ocr
```

命令输出：

```json
[
  [
    "***",
    0.9297130703926086
  ],
  [
    "姓名",
    0.9741479158401489
  ],
  [
    "别男民族汉",
    0.991115391254425
  ],
  [
    "生",
    0.9976730942726135
  ],
  [
    "19**年*月*日",
    0.9517846703529358
  ],
  [
    "住址",
    0.9953198432922363
  ],
  [
    "**市**区**路**",
    0.9569936394691467
  ],
  [
    "******-*-****",
    0.9409841299057007
  ],
  [
    "公民身份号码",
    0.9767408967018127
  ],
  [
    "******************",
    0.9046091437339783
  ]
]
```

数组内的第一个字段是识别的文字内容，第二个字段是置信值。

也可以在浏览器中访问 http://localhost:8080/v1/img/ocr 上传图片文件进行测试。

## 六、执行生产部署

虽然**Flask 的内建服务器**轻便且易于使用，但是 **Flask 的内建服务器不适用于生产** ，它也不能很好的扩展。由于 Flask 应用支持[Gunicorn](https://dormousehole.readthedocs.io/en/latest/deploying/wsgi-standalone.html#gunicorn)、[uWSGI](https://dormousehole.readthedocs.io/en/latest/deploying/wsgi-standalone.html#uwsgi)、[Gevent](https://dormousehole.readthedocs.io/en/latest/deploying/wsgi-standalone.html#id3)、[Eventlet](https://dormousehole.readthedocs.io/en/latest/deploying/wsgi-standalone.html#id4)、[Twisted Web](https://dormousehole.readthedocs.io/en/latest/deploying/wsgi-standalone.html#twisted-web)等多种方式自主部署，以下选用 uWSGI 执行自主部署。

[uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/) 一个用 C 编写的快速应用服务器。它配置丰富，也为撰写强大的网络应用提供了许多其他工具。 告诉 uWSGI 如何导入你的 Flask 应用对象就可以运行 Flask 应用。

> 请务必把 `app.run()` 放在 `if __name__ == '__main__':` 内部或者放在单独的文件中，这样可以保证它不会被调用。因为，每调用一次就会开启一个本地 WSGI 服务器。当我们使用 uWSGI 部署应用时，不需要使用本地服务器。

安装uwsgi

```shell
python3 -m pip install uwsgi
```

命令输出：

```shell
Collecting uwsgi
  Downloading uwsgi-2.0.21.tar.gz (808 kB)
     |████████████████████████████████| 808 kB 806 kB/s 
Building wheels for collected packages: uwsgi
  Building wheel for uwsgi (setup.py) ... done
  Created wheel for uwsgi: filename=uWSGI-2.0.21-cp37-cp37m-linux_x86_64.whl size=559887 sha256=c02418d94313937f621fc9fbac7ae073a349a11f436661164bdefba01669774e
  Stored in directory: /root/.cache/pip/wheels/b1/b8/6a/cafb5a30fed7e484147b84224e4264ab3930dfaf0586c326fb
Successfully built uwsgi
Installing collected packages: uwsgi
Successfully installed uwsgi-2.0.21
```

将正式发布的 paddleocr_http.py 文件从外部挂载目录 `/paddle` 移动到 Docker 容器内部 `/home` 目录 

```shell
cp -rp /paddle/paddleocr_http.py /home/paddleocr_http.py
```

uWSGI 提供包括 HTTP/HTTPS router/proxy/load-balancer 多种前置服务模式。本次我们选择 HTTP 模式，在使用 uWSGI 的 HTTP 服务时，uWSGI 也是将请求转发给 uWSGI 工作者，并提供了两种方式：嵌入式和独立式。在嵌入式模式下，它将自动生成 uWSGI 工作者并设置通信套接字。在独立模式下，你必须指定要连接的uWSGI套接字的地址。我们选择嵌入模式。

uwsgi 是基于 python 模块中的 WSGI 调用的。我们的 Flask 应用名称为 paddleocr_http.py ， 可以使用以下命令：

```shell
uwsgi --http 0.0.0.0:8080 --master --wsgi-file /home/paddleocr_http.py --callable app --processes 4 --threads 2 
```

参数 `-p 4` 表示一次最多可以使用 4 个 worker 来处理 4 个请求。 `--http 0.0.0.0:8080` 表示在所有接口的 8080 端口上提供服务。

命令输出：

```shell
*** Starting uWSGI 2.0.21 (64bit) on [Tue Jan 31 02:22:59 2023] ***
compiled with version: 7.5.0 on 31 January 2023 01:58:47
os: Linux-5.15.49-linuxkit #1 SMP Tue Sep 13 07:51:46 UTC 2022
nodename: 874df41b5721
machine: x86_64
clock source: unix
detected number of CPU cores: 6
current working directory: /paddle
detected binary path: /usr/local/bin/uwsgi
!!! no internal routing support, rebuild with pcre support !!!
uWSGI running as root, you can use --uid/--gid/--chroot options
*** WARNING: you are running uWSGI as root !!! (use the --uid flag) *** 
your memory page size is 4096 bytes
detected max file descriptor number: 1048576
lock engine: pthread robust mutexes
thunder lock: disabled (you can enable it with --thunder-lock)
uWSGI http bound on 0.0.0.0:8080 fd 4
uwsgi socket 0 bound to TCP address 127.0.0.1:36529 (port auto-assigned) fd 3
uWSGI running as root, you can use --uid/--gid/--chroot options
*** WARNING: you are running uWSGI as root !!! (use the --uid flag) *** 
Python version: 3.7.13 (default, Apr 24 2022, 01:04:09)  [GCC 7.5.0]
Python main interpreter initialized at 0x555a043df120
uWSGI running as root, you can use --uid/--gid/--chroot options
*** WARNING: you are running uWSGI as root !!! (use the --uid flag) *** 
python threads support enabled
your server socket listen backlog is limited to 100 connections
your mercy for graceful operations on workers is 60 seconds
mapped 416880 bytes (407 KB) for 8 cores
*** Operational MODE: preforking+threaded ***
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
WSGI app 0 (mountpoint='') ready in 2 seconds on interpreter 0x555a043df120 pid: 1145 (default app)
uWSGI running as root, you can use --uid/--gid/--chroot options
*** WARNING: you are running uWSGI as root !!! (use the --uid flag) *** 
*** uWSGI is running in multiple interpreter mode ***
spawned uWSGI master process (pid: 1145)
spawned uWSGI worker 1 (pid: 1165, cores: 2)
spawned uWSGI worker 2 (pid: 1168, cores: 2)
spawned uWSGI worker 3 (pid: 1171, cores: 2)
spawned uWSGI worker 4 (pid: 1174, cores: 2)
spawned uWSGI http 1 (pid: 1177)
```

这里我们忽略了 root 模式运行的警告信息。

由于 Docker 中不能使用 systemd 进行服务自启动，为了遵循一个 docker 容器进程运行一个服务的规范，准备以paddlepaddle/paddle:2.4.1为基础镜像，创建paddle-http的镜像。如果是有CUDA、ROCm的运行环境，记得修改对应的基础镜像创建 paddle-http的镜像。

首先在本机（非 Docker 容器中） `paddleocr_http.py` 相同的目录下编写 `Dockerfile`，内容如下：

```dockerfile
# 基于paddleocr镜像，注意选择需要的版本号以及CPU、GPU类型
FROM paddlepaddle/paddle:2.4.1

# 设置工作目录为 /demo
WORKDIR /home

# 将依赖文件拷贝到工作目录
COPY paddleocr_http.py /home
COPY test.jpg /home

# 执行pip指令，安装这个应用所需要的依赖，当前只安装了paddleocr模型，和运行需要的uwsgi服务
# 如果需要在本镜像内使用更多模型，可在此处添加，并增加 paddleocr_http.py 程序功能
RUN python3 -m pip install paddleocr uwsgi
# 执行一次测试，让paddleocr下载模型库
RUN paddleocr --image_dir ./test.jpg --use_angle_cls true --use_gpu false --lang=ch

# 允许外界访问8080端口
EXPOSE 8080

# 设置容器进程为uwsgi嵌入模式启动
ENTRYPOINT ["uwsgi", "--http", "0.0.0.0:8080", \
    "--master", \
    "--wsgi-file", "/home/paddleocr_http.py", \
    "--callable", "app", \
    "--processes", "4", \
    "--threads", "2"]
```

修改 paddleocr_http.py 文件，设置 Flask 默认日志级别为INFO

```python
"root": {"level": "INFO", "handlers": ["wsgi"]},
```

执行 Docker 编译

```shell
docker build . -t paddle-http
```

命令输出：

```shell
[+] Building 268.6s (11/11) FINISHED                                                  
=> [internal] load build definition from Dockerfile                            0.0s  
=> => transferring dockerfile: 37B                                             0.0s
=> [internal] load .dockerignore                                               0.0s
=> => transferring context: 2B                                                 0.0s
=> [internal] load metadata for docker.io/paddlepaddle/paddle:2.4.1            1.2s  
=> [internal] load build context                                               0.0s  
=> => transferring context: 180B                                               0.0s  
=> [1/6] FROM docker.io/paddlepaddle/paddle:2.4.1@sha256:72d9cfad34dcfae39743  0.0s  
=> CACHED [2/6] WORKDIR /home                                                  0.0s  
=> CACHED [3/6] COPY paddleocr_http.py /home                                   0.0s  
=> CACHED [4/6] COPY test.jpg /home                                            0.0s  
=> [5/6] RUN python3 -m pip install paddleocr uwsgi                          237.8s  
=> [6/6] RUN paddleocr --image_dir ./test.jpg --use_angle_cls true --use_gpu  22.0s  
=> exporting to image                                                          7.5s
=> => exporting layers                                                         7.5s  
=> => writing image sha256:a4469cddbb63ffa66c25826a96a7a912e13107a1811dd665e3  0.0s 
=> => naming to docker.io/library/paddle-http                                  0.0s 
                                                                                     
Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
```

停止 `paddle_docker` docker 容器，防止端口占用

```shell
docker stop paddle_docker
```

第一次启动 `paddle-http` 应用

```shell
docker run --name paddle-http -p 8080:8080 paddle-http
```

停止后再次启动

```shell
docker start paddle-http
```

paddle-http启动成功后，可以通过浏览器访问 http://localhost:8080/v1/img/ocr 执行测试。

## 七、参照资料

1. [Flask](https://flask.palletsprojects.com/en/2.2.x/)
2. [uWSGI Deploying Flask](https://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html)
3. [PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/macos-docker.html)
4. [PaddleOCR网址](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/quickstart.md)
5. [Dockerfile资料](https://docs.docker.com/engine/reference/builder/)
