import gradio as gr
import zipfile
from glob import glob
import os
from PIL import Image
from collections import Counter

from huggingface_hub import hf_hub_download, login 
login(token=os.getenv('LOGIN_TOKEN'))
hf_hub_download(repo_id="giniwini/model_creator", filename="ModelCreator.py", local_dir='.')

from ModelCreator import Model

m = Model()

markdown_head = """# <center>🚀👁️ Create your own Image Classifier 👁️🚀</center>      

Have you got a personal project that needs a tool to automatically classify images? \n
This space is intended to give high level tools so everyone can make his own image classification model and use it for any purpose. 

## How to create it and test it

1. Put some images in a folder classified by subfolders indicating the classes. Like: \n
				- img_folder/cat/image_of_cat_0.png
				- img_folder/dog/image_of_dog_4.jpeg \n
I recommend around 5 images per class. \n
2. Right click in the folder and press "compress..." in ".zip" mode. This will create you a zip file. \n
3. Upload the zip file on this space and press "Sumbit Zip File" button. \n
4. Congratulations you already have the model. Try it by uploading some images on 'input image' and press "Predict/Test on an image". \n  
5. Do you want to export the model to use It outside this space? You will need a password. Check the information at the bottom ⬇️ 

---
"""

markdown_tail = """
---

## Are you grateful?

Consider buying me a coffee subscribing to my [patreon](https://patreon.com/elokquentia?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink) coffee tier \n
or make a donation trough [paypal](https://www.paypal.com/donate/?hosted_button_id=QD2W2G34GWQ4J)

---

## Export the model
Subscribe to patreon Brunch tier to get the password. [patreon](https://patreon.com/elokquentia?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink)

---
## Have you already exported the model? Here's how to use it

### 1. Install the requirements: 
Open the terminal and go to the directory where you placed the requirements.txt file. \n 
Now with this command you can install the dependencies at once. 
```bash
pip install -r requirements.txt
```
Has anything gone wrong? Look inside the requirements.txt and install the dependencies one by one. 

### 2. Use ModelCreator's Model class
Place the ModelCreator.py file on the directory you want to code. 
    - some_folder/ModelCreator.py
    - your_python_script.py

### 3. Use your model
In your_python_script.py, try something like:
```python
from ModelCreator import Model

model_path = 'model.pickle'
m2 = Model().load_model(model_path)

image_path = 'some_image_path/image_1.png'
result = m2(image_path)
print(result)
```
### 4. DO YOU HAVE ANY DOUBTS ON HOW THIS WORKS OR PROBLEMS USING THE MODEL?
Contact me on huggingface or via e-mail genlain@gmail.com
"""


def fit_model(zip_file_path):
    path = 'tmp/'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    images_path = glob(f'{path}**/*/*.*')
    labels_train = [i.split('/')[-2] for i in images_path]
    print(labels_train)
    images_train = [Image.open(i) for i in images_path]

    m.train(images_train, labels_train)
    return (f"Model Fitted \n"
            f"Classes: {dict(Counter(labels_train))}")


def predict(image):
    l = m.predict(image)
    return f'Predicted Class: {l[0]}'


def export(password):
    if password == os.getenv('EXPORT_PASSWORD'):
        m.export_model()
        outs = []
        for file in [f'model.pickle', f'requirements.txt', f'ModelCreator.py']:
            outs += [gr.update(visible=True), file]
        return outs
    else:
        return [None, f'Subscribe to Patreon to Download']*3

with gr.Blocks() as demo:
    gr.Markdown(markdown_head)

    with gr.Row() as g1:
        inp = gr.File(label="zip file")
        out = gr.Textbox(label='Message')
    btn = gr.Button("Submit Zip File")
    btn.click(fn=fit_model, inputs=inp, outputs=out)

    with gr.Row() as g2:
        inp2 = gr.Image(label="Input Image", type='pil')
        out2 = gr.Textbox(label='Prediction')
    btn2 = gr.Button("Predict/Test on an Image")
    btn2.click(fn=predict, inputs=inp2, outputs=out2)

    with gr.Row() as g3:
        inp3 = gr.Textbox(label='Password to Download')
        out3 = gr.File(label='Model Download link', visible=True, height=30, interactive=False)
        out4 = gr.File(label='Requirements Download link', visible=True, height=30, interactive=False)
        out5 = gr.File(label='ModelCreator.py Download link', visible=True, height=30, interactive=False)
    btn3 = gr.Button("Export Fitted Model")
    btn3.click(fn=export, inputs=inp3, outputs=[out3, out3, out4, out4, out5, out5])
    gr.Markdown(markdown_tail)
demo.launch()
