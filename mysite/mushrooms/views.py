from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps
import tensorflow as tf
import os

classes = ['съедобный', 'несъедобный']

model = tf.keras.models.load_model('mushrooms/mushrooms_final.h5')

def index(request):
    if request.method == 'POST' and request.FILES:
        file = request.FILES['myfile']
        upload_folder = os.path.join('mushrooms', 'uploaded_images')
        fs = FileSystemStorage(location=upload_folder)

        filename = fs.save(file.name, file)
        file_url = fs.url(filename)


        size = (180, 180)
        image = Image.open(file)
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array).flatten()
        predictions = tf.where(predictions < 0.5, 0, 1)
        predicted_class_index = tf.argmax(predictions)

        res_message = f'Скорее всего на этом фото {classes[predicted_class_index]} гриб'

        return render(request, 'mushrooms/index.html', {
            'res_message': res_message,
        })
    return render(request, 'mushrooms/index.html')