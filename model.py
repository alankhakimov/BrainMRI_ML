import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def train_model():
    # --- Parameters ---
    base_dir = r"C:\\Documents\\MRI Machine Learning Project\\3 category brain MRI\\Brain_Cancer raw MRI data\\Split_Dataset"
    img_size = (128, 128)
    batch_size = 32
    class_names = ['Glioma', 'Menin', 'Cancer']

    # --- Load Datasets ---
    train_ds = image_dataset_from_directory(os.path.join(base_dir, 'train'), image_size=img_size, batch_size=batch_size)
    val_ds = image_dataset_from_directory(os.path.join(base_dir, 'val'), image_size=img_size, batch_size=batch_size)
    test_ds = image_dataset_from_directory(os.path.join(base_dir, 'test'), image_size=img_size, batch_size=batch_size)

    # --- Normalize ---
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # --- Initial Training ---
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train frozen base
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    # --- Fine-Tuning ---
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Freeze initial layers if desired
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5)

    # ✅ Save the model
    model_path = "C:\\Documents\\MRI Machine Learning Project\\3 category brain MRI\\model.keras"
    model.save(model_path)

    # --- Evaluate on Test Set ---
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test accuracy:", test_acc)

    # --- Plot Training Curve ---
    plt.plot(history.history['accuracy'], label='Train Accuracy (initial)')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy (initial)')
    plt.plot(history_fine.history['accuracy'], label='Train Accuracy (fine-tuned)')
    plt.plot(history_fine.history['val_accuracy'], label='Val Accuracy (fine-tuned)')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # --- Confusion Matrix ---
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.colorbar()
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.show()

# --- Classify New Image Function ---
def classify_image(img_path, model_path):
    model = load_model(model_path)
    img_size = (128, 128)
    class_names = ['Glioma', 'Menin', 'Cancer']
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    return class_names[predicted_class], confidence

def main():
    img_path = Path("C:\Documents\MRI Machine Learning Project\\3 category brain MRI\Brain_Cancer raw MRI data\Split_"
                    "Dataset\\test\\brain_glioma\\brain_glioma_0728.jpg")
    classification = classify_image(img_path, "C:\\Documents\\MRI Machine Learning Project\\3 category brain MRI\\model.keras")
    print(classification)

if __name__ == '__main__':
    main()