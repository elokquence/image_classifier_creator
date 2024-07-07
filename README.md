# <center> Create your own Image Classifier </center>  

* Here's a part of the project I have done so anyone can build its own Image Classifier in seconds. 
I have deployed it in [HuggingFace Space](https://huggingface.co/spaces/elokquence37/classifier_creator)
You can use It and even pay me if you are grateful. Feel free to directly use the class Model on the ModelCreator.py file, and change it if necessary.
---
---

# <center> ⬇️  Deployed in HuggingFace   ⬇️ </center> 
## <center>🚀👁️ Create your own Image Classifier 👁️🚀</center>      
Have you got a personal project that needs a tool to automatically classify images? \n
This space is intended to give high level tools so everyone can make his own image classification model and use it for any purpose. 
### How to create it and test it
1. Put some images in a folder classified by subfolders indicating the classes. Like: \n
				- img_folder/cat/image_of_cat_0.png
				- img_folder/dog/image_of_dog_4.jpeg \n
I recommend around 5 images per class. \n
2. Right click in the folder and press "compress..." in ".zip" mode. This will create you a zip file. \n
3. Upload the zip file on this space and press "Sumbit Zip File" button. \n
4. Congratulations you already have the model. Try it by uploading some images on 'input image' and press "Predict/Test on an image". \n  
5. Do you want to export the model to use It outside this space? You will need a password. Check the information at the bottom ⬇️ 
---


---
### Are you grateful?

* Consider buying me a coffee subscribing to my [patreon](https://patreon.com/elokquentia?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink) coffee tier \n
or make a donation trough [paypal](https://www.paypal.com/donate/?hosted_button_id=QD2W2G34GWQ4J)
---
### Export the model

* Subscribe to patreon Brunch tier to get the password. [patreon](https://patreon.com/elokquentia?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink)
---
### Have you already exported the model? Here's how to use it
#### 1. Install the requirements: 

Open the terminal and go to the directory where you placed the requirements.txt file. \n 
Now with this command you can install the dependencies at once. 
```bash
pip install -r requirements.txt
```
Has anything gone wrong? Look inside the requirements.txt and install the dependencies one by one. 
#### 2. Use ModelCreator's Model class
Place the ModelCreator.py file on the directory you want to code. 
    - some_folder/ModelCreator.py
    - your_python_script.py
#### 3. Use your model
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
