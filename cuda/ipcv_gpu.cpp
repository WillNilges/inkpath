#include "ipcv_gpu.h"

// Skeletonization algorithm. I might mess around with this
// more down the road.
// TODO: Where the hell did I find this?
Mat gpu_skeletonize(Mat img_inv, std::string output_path) {

    cv::cuda::Stream stream1;

    Mat img;
    bitwise_not(img_inv, img);
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
     
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
     
    bool done;		
    do
    {
      cv::erode(img, eroded, element);
      cv::dilate(eroded, temp, element); // temp = open(img)
      cv::cuda::subtract(img, temp, temp, cv::noArray(), -1, stream1);
      cv::cuda::bitwise_or(skel, temp, skel, cv::noArray(), stream1);
      eroded.copyTo(img);
     
      done = (cv::countNonZero(img) == 0);
    } while (!done);

    Mat skel_invert;
    cv::cuda::bitwise_not(skel, skel_invert, cv::noArray(), stream1);

    //Mat downsampled;
    //pyrDown(skel_invert, downsampled, Size( img.cols/2, img.rows/2 ));
    if (!output_path.empty()) {
        imwrite(output_path, skel_invert);
#ifdef DIAG
        std::cout << "Image has been written to " << output_path << "\n";
#endif
    }
    return skel_invert;
}

// Apply an Otsu's thresholding to the object. I found that this was
// the best function of the ones I tried
Mat gpu_otsu(Mat img, std::string output_path)
{
    cv::cuda::Stream stream1;

    int k;
    // Upsample our image, if needed.
    Mat upsampled;
    cv::cuda::GpuMat gpu_pre, gpu_upsampled;
    if (img.rows < 1000 || img.cols < 1000) {
        gpu_pre.upload(img);
        cv::cuda::pyrUp(gpu_pre, gpu_upsampled, stream1);
        gpu_upsampled.download(upsampled);
    } else {
        upsampled = img;
    }

    //otsu's thresholding after gaussian filtering
    Mat gauss_thresh;
    Mat blur;
    cv::cuda::GpuMat gpu_blur_in, gpu_blur;
    //GaussianBlur(upsampled, blur, Size(5, 5), 0, 0); 
    cv::Ptr<cv::cuda::Filter> gauss_filter = cv::cuda::createGaussianFilter(upsampled.type(), -1, Size(5, 5), 0, 0);
    gpu_blur_in.upload(upsampled); // Pass to GPU
    gauss_filter->apply(gpu_blur_in, gpu_blur, stream1);
    gpu_blur.download(blur); // Back to CPU
    // TODO: Implement my own thresholding
    threshold(blur, gauss_thresh, 0, 255, THRESH_OTSU); 
    //adaptiveThreshold(blur, gauss_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 2); 

    if (!output_path.empty()) {
        imwrite(output_path, gauss_thresh);
#ifdef DIAG
        std::cout << "Image has been written to " << output_path << "\n";
#endif
    }
    return gauss_thresh;
}

// Prereqs: Must be binary color image, target must be black
Shapes gpu_find_shapes(Mat img, std::string output_path) {
    Mat src;
    cv::cuda::Stream stream1;
    cv::cuda::bitwise_not(img, src, cv::noArray(), stream1);

    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    src = src > 1;
    //namedWindow( "Source", 1 );
    //imshow( "Source", src );
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( src, contours, hierarchy,
        RETR_TREE, CHAIN_APPROX_SIMPLE );

    if (!output_path.empty()) {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            Scalar color( rand()&255, rand()&255, rand()&255 );
            drawContours( dst, contours, idx, color, FILLED, 8, hierarchy );
        }

        imwrite(output_path, dst);
#ifdef DIAG
        std::cout << "Image has been written to " << output_path << "\n";
#endif
    }

    return Shapes{contours, hierarchy};
}
