#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

//把图像归一化为0-255，便于显示
cv::Mat norm_0_255(const cv::Mat& src)
{
    cv::Mat dst;
    switch(src.channels())
        {
    case 1:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
        }
    return dst;
}

//转化给定的图像为行矩阵
cv::Mat asRowMatrix(const std::vector<cv::Mat>& src, int rtype, double alpha = 1, double beta = 0)
{
    //样本数量
    size_t n = src.size();
    //如果没有样本，返回空矩阵
    if(n == 0)
        return cv::Mat();
    //样本的维数
    size_t d = src[0].total();

    cv::Mat data(n, d, rtype);
    //拷贝数据
    for(int i = 0; i < n; i++)
        {

        if(src[i].empty()) 
            {
            std::string error_message = cv::format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
            }
        // 确保数据能被reshape
        if(src[i].total() != d) 
            {
            std::string error_message = cv::format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
            }
        cv::Mat xi = data.row(i);
        //转化为1行，n列的格式
        if(src[i].isContinuous())
            {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
            } else {
                src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
            }
        }
    return data;
}

int main(int argc, const char *argv[])
{

    std::vector<cv::Mat> db;

    std::string prefix = "/home/liu/图片/projects/face/";
    
    int Q = 9;
    for(int i=0;i<Q;i++)
    {
        std::string fileName = prefix + std::to_string(i)+".jpg";
        cv::Mat src = cv::imread(fileName,0);
        cv::resize(src,src,cv::Size(100,100));
        db.push_back(src);
    }

    // Build a matrix with the observations in row:
    cv::Mat data = asRowMatrix(db, CV_32FC1);//Q*MN

    // PCA算法保持5主成分分量
    int num_components = 9;

    //执行pca算法
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //pca算法结果
    cv::Mat mean = norm_0_255(pca.mean.reshape(1, db[0].rows));
    cv::Mat eigenvalues = pca.eigenvalues.clone();
    cv::Mat eigenvectors = pca.eigenvectors.clone();//p*MN
    
    std::cout<<"eigenvalues:\n"<<eigenvalues<<std::endl;
    std::cout<<eigenvectors.size<<std::endl;

    //均值脸
    cv::imshow("avg", mean);

    //五个特征脸
    for(int i=0;i<num_components;i++)
    {
        std::string winName = "pc" + std::to_string(i);
        cv::namedWindow(winName,0);
        cv::imshow(winName, norm_0_255(pca.eigenvectors.row(i)).reshape(1, db[0].rows));
    }
    
    //投影到特征脸空间
    for(int i=0;i<Q;i++)
    {
        cv::Mat dataLine = data(cv::Range(i,i+1),cv::Range(0,data.cols));
        dataLine -= pca.mean;
    }
    cv::Mat om = data*eigenvectors.t();//Q*p
    std::cout<<"om:\n"<<om<<std::endl;
    
    //由特征空间重建
    /*cv::Mat b = om.row(0)*eigenvectors+pca.mean;
    cv::Mat nb = norm_0_255(b.reshape(1,db[0].rows));
    cv::namedWindow("nb",0);
    cv::imshow("nb",nb);
    */
    
    //识别
    std::vector<cv::Mat> Is(1);
    Is[0] = cv::imread("/home/liu/图片/projects/face/9.jpg",0);
    cv::resize(Is[0],Is[0],cv::Size(100,100));
    cv::Mat Idata = asRowMatrix(Is,CV_32FC1);
    cv::Mat Iom = (Idata-pca.mean)*eigenvectors.t();
    std::cout<<"Iom:\n"<<Iom<<std::endl;
    for(int i=0;i<Q;i++)
    {
        double epsilon = cv::norm(Iom-om.row(i));
        std::cout<<epsilon<<std::endl;
    }

    cv::waitKey();

    // Success!
    return 0;
}

