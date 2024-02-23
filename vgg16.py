import tensorflow as tf
from tensorflow.keras import optimizers,losses,models,datasets,Sequential
from tensorflow.keras.layers import Dense,Conv2D,BatchNormalization,MaxPooling2D,Flatten
 
 
class vgg16(models.Model):
    def __init__(self):
        super(vgg16, self).__init__()
        self.model = models.Sequential([
            Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'),
            Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Conv2D(filters=128,kernel_size = (3,3),padding='same',activation='relu'),
            Conv2D(filters=128,kernel_size = (3,3),padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Conv2D(filters=256, kernel_size=(3, 3),padding='same', activation='relu'),
            Conv2D(filters=256, kernel_size=(3, 3),padding='same', activation='relu'),
            Conv2D(filters=256, kernel_size=(3, 3),padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Conv2D(filters=512, kernel_size=(3, 3),padding='same', activation='relu'),
            Conv2D(filters=512, kernel_size=(3, 3), padding='same',activation='relu'),
            Conv2D(filters=512, kernel_size=(3, 3), padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Conv2D(filters=512, kernel_size=(3, 3), padding='same',activation='relu'),
            Conv2D(filters=512, kernel_size=(3, 3),padding='same', activation='relu'),
            Conv2D(filters=512, kernel_size=(3, 3), padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Flatten(),
            Dense(512,activation='relu'),
            Dense(256,activation='relu'),
            Dense(10,activation='softmax')
        ])
 
    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x
 
 
def main():
    (train_x,train_y),(test_x,test_y) = datasets.cifar10.load_data()
    train_x = train_x.reshape(-1,32,32,3) / 255.0
    test_x = test_x.reshape(-1,32,32,3) / 255.0
 
    model = vgg16()
 
    # model.build((None,32,32,3))
    # model.summary() 不使用类写VGG的话，就不报错，使用了类写VGG就报错，我也很无奈
 
    model.compile(optimizer=optimizers.Adam(0.01),
                  loss = losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_x,train_y,epochs=10,batch_size=256)
 
    score = model.evaluate(test_x,test_y,batch_size=50)
 
    print('loss:',score[0])
    print('acc:',score[1])
    pass
 
 
if __name__ == '__main__':
    main()
