# Comparing Knn and Decision Tree Performance 

### This project is done on a German Credit Data, identifying if someone is a foreign worker based on 24 attributes such as status of existing checking account, credit history, purpose for credit (car, travel, education, business, etc), employer status, present residence since, property type, age,  and so on. Dataset contains 1000 instances and it already numeric.

Here I am comparing KNN and Decision tree based on F-measure and time it takes to predict outputs using our model. I am running both models in 5 iterations and then I find the average. Additionally, I try out different k and maxDepth for Knn and DT respectively. Finally, I compare output results by creating barcharts using matplotlib, as well as numeric values in the console. 
Some plots from the code:
![image](https://user-images.githubusercontent.com/60479692/190829660-dc911c64-934b-4bba-8585-8eb5fc579446.png)

![image](https://user-images.githubusercontent.com/60479692/190829702-988f80ec-969e-40e9-8ecf-1263f2637474.png)
<sub> As clearly seen above, Knn is slightly more accurate than Decision Trees, however, Knn takes much greater time to make its prediction. As such, decision trees would be optimal despite its  lacking accuracy for quick rapid predictions.  </sub>

![image](https://user-images.githubusercontent.com/60479692/190829779-baedaad1-2d00-4b5c-be36-bc28bd521947.png)
<sub> </sub> Final comparison by F-measure when Knn model uses the best k and Decision Tree model uses the best max_depth
