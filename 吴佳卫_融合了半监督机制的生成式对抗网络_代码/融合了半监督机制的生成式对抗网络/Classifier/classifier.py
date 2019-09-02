import os
import model_train
import model_test

CycleGAN_Train = True #设置为True即处理CycleGAN的训练集，设置为False即处理CycleGAN的测试集
Data_A = True #设置为True即处理A类图像，设置为False即处理B类图像
Run_Train = True #设置为True即训练CNN分类器，设置为False即运行CNN分类器


def main():
    learning_rate = 1e-4
    if CycleGAN_Train is True:
        if Data_A is True:
            train = os.path.realpath(os.curdir + "./Data/Data_processed/A/train")
            test = os.path.realpath(os.curdir + "./Data/Data_processed/A/test")
            output0 = os.path.realpath(os.curdir + "./Data/Data_classified/A/0")
            output1 = os.path.realpath(os.curdir + "./Data/Data_classified/A/1")
            save = os.path.realpath(os.curdir + "/save_net/A/A_checkpoint.ckpt")
        else:
            train = os.path.realpath(os.curdir + "./Data/Data_processed/B/train")
            test = os.path.realpath(os.curdir + "./Data/Data_processed/B/test")
            output0 = os.path.realpath(os.curdir + "./Data/Data_classified/B/0")
            output1 = os.path.realpath(os.curdir + "./Data/Data_classified/B/1")
            save = os.path.realpath(os.curdir + "/save_net/B/B_checkpoint.ckpt")

        if Run_Train is True:
            model_train.conv_net_model_train(learning_rate, train, save)
        else:
            model_test.conv_net_model_test(save, test, output0, output1)

    else:
        if Data_A is True:
            train = os.path.realpath(os.curdir + "./Data/Data_processed/A/train")
            test = os.path.realpath(os.curdir + "./Data/Data_test/processed_data/A")
            output0 = os.path.realpath(os.curdir + "./Data/Data_test/cycle_GAN_result/A")
            output1 = os.path.realpath(os.curdir + "./Data/Data_test/classified_data/A")
            save = os.path.realpath(os.curdir + "/save_net/A/A_checkpoint.ckpt")
        else:
            train = os.path.realpath(os.curdir + "./Data/Data_processed/B/train")
            test = os.path.realpath(os.curdir + "./Data/Data_test/processed_data/B")
            output0 = os.path.realpath(os.curdir + "./Data/Data_test/cycle_GAN_result/B")
            output1 = os.path.realpath(os.curdir + "./Data/Data_test/classified_data/B")
            save = os.path.realpath(os.curdir + "/save_net/B/B_checkpoint.ckpt")
            
        if Run_Train is True:
            model_train.conv_net_model_train(learning_rate, train, save)
        else:
            model_test.conv_net_model_test(save, test, output0, output1)


if __name__ == "__main__":
    main()
