# 基于paddle镜像，注意选择需要的版本号以及CPU、GPU类型
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