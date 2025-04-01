with gr.Blocks() as demo:  # web页面效果主代码

    with gr.Tab("药典问答"):  # 问答页面
        # 创建 Chatbot 实例并设置高度
        gr.Markdown("**第一次使用请先去配置自己的信息并保存哦**")
        chatbot = gr.Chatbot(height=600)  # 设置高度为600像素
        qa_interface = gr.ChatInterface(
            fn=slow_echo,
            chatbot=chatbot  # 将自定义的 Chatbot 传递给 ChatInterface
        )

    with gr.Tab("文档导入"):  # 文档导入页面
        folder_input = gr.Textbox(label="保存向量数据库的文件夹路径", value=r"C:/Users/abc10/桌面/workllm/ragproject")

        upload_interface = gr.Interface(
            fn=import_new_documents,
            inputs=[
                gr.File(label="上传新的药典文档"),
                gr.Textbox(label="数据库和索引名命名", placeholder="请输入"),
                folder_input,
            ],
            outputs=gr.Textbox(label="结果"),
            title="文档导入",
            description="上传新的药典文档以更新数据库。"
        )

    with gr.Tab("配置"):  # 配置页面
        es_host_input = gr.Textbox(label="ES主机地址", value='192.168.110.28')
        es_port_input = gr.Textbox(label="ES服务端口", value='9200')
        es_user_input = gr.Textbox(label="ES用户名", value='elastic')
        es_pass_input = gr.Textbox(label="ES密码", type="password", value='7ztvwEMjr0H+_R4Vec*R')

        es_index_input = gr.Textbox(label="ES索引名（上传的会和向量数据库同名）", value='zhyd')  # 新增索引名输入框
        vector_db_path_input = gr.Textbox(label="向量数据库位置",
                                          value='C:/Users/abc10/桌面/workllm/ragproject/embeddings2.npz')  # 新增向量数据库位置输入框

        config_submit = gr.Button("保存配置")
        config_message = gr.Textbox(label="状态", interactive=False)

        config_submit.click(
            fn=update_config,
            inputs=[es_host_input, es_port_input, es_user_input, es_pass_input, es_index_input, vector_db_path_input],
            outputs=config_message
        )