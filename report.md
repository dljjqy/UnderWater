# How predict the index of water quality by networks

## Clean and scale data
For example,in the water quality index of WangFei island,there are fiftten indicators(采集时间, 接收时间, 采集方式, 水温, pH, 溶解氧, 电导率, 浊度, 高锰酸盐指数, 氨氮, 总磷, 总氮, 湿度, 室温, 叶绿素, 藻密度).However, we do not care about all the fifteen indicators and some indicators have too little data to use.The first thing we need to do is select some indicators we care about most.Here we choose [采集时间, 水温, pH, 溶解氧, 电导率, 浊度, 高锰酸盐指数, 氨氮, 总磷, 总氮],total nine indicators except the time index.
* 采集时间: Acquisition time.
* 水温: Water temperture.
* 电导率: Conductivity
* 浊度: Turbidity.
* 高锰酸盐指数: Permanganate index.
* 氨氮: Ammonia nitrogen.
* 总磷: Total phosphorus(TP).
* 总氮: Total nitrogen(TN).

After checking the numbers in data,we could find there are many outliers between those numbers.We can see it more clearly by image.
![Original Data](/images/origional.png "Original Data")

First we want to remove the numbers which are absolutely wrong.For example,the numbers of conductivity should be range from 50 to 500 uS/cm,as the data comes from a river.Below we show the common range of the nine index.
* Water temperture: (5, 30)
* pH: (5.0, 9.0)
* Dissolved Oxygen: (1, 15)
* Conductivity: (50, 500)
* Turbidity: (0, 1500)
* Permanganate Index: (0, 15)
* AN: (0, 0.5)
* TP: (0, 0.3)
* TN: (0, 5)

Below is the figure of the data has been cutoff.
![Cutoff Data](/images/cutoff.png "Cutoff Data")

After we cutoff the wrong numbers which coming from measurement mistakes, there are still many outliers which may be generated from measurement error.We mainlu use Zscore and Standard Deviation mthods.[Outlier Methods](https://towardsdatascience.com/outlier-detection-part1-821d714524c#:~:text=For%20example%2C%20a%20z%2Dscore,similar%20to%20standard%20deviation%20method.).


