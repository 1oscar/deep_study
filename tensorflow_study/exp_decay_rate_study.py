#coding:utf-8
#设损失函数 loss=(w+1)^2，令w初值是常数5，反向传播就是求最优w,即最小loss对应的w值
import tensorflow as tf

LEARNING_RATE_BASE=0.1    #最初学习率
LEARNING_RATE_DECAY=0.99  #学习率衰减率
LEARNING_RATE_STEP=1      #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE

#运行了几轮BATCH_SIZE的计数器，初值给0，设为不被训练
global_step=tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
'''
其中， LEARNING_RATE_BASE 为学习率初始值， LEARNING_RATE_DECAY 为学习率衰减率,global_step 记录了当前训练轮数，为不可训练型参数。学习率 learning_rate 更新频率为输入数据集总样本数除以每次喂入样本数。若 staircase 设置为 True 时，表示 global_step/learning rate step 取整数，学习率阶梯型衰减；若 staircase 设置为 false 时，学习率会是一条平滑下降的曲线。
例如：在本例中，模型训练过程不设定固定的学习率，使用指数衰减学习率进行训练。其中，学习率初值设置为 0.1，学习率衰减率设置为 0.99， BATCH_SIZE 设置为 1
learning_rate=learning_rate_base*learning_rate_decay*(global_step/learning_rate_batch_size)
'''
#定义待优化参数w初值赋5
w=tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数loss
loss=tf.square(w+1)
#定义反向传播方法
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#生成会话，训练40轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val=sess.run(learning_rate)
        global_step_val=sess.run(global_step)
        w_val=sess.run(w)
        loss_val=sess.run(loss)
        print("After %s steps: global step is %f,w is %f,learning rate is %f loss is %f."%(i,global_step_val,w_val,learning_rate_val,loss_val))
