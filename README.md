Download Link: https://assignmentchef.com/product/solved-cs433-exercise-2-linear-regression-and-gradient-descent
<br>
Linear Regression and Gradient Descent

Goals.       The goal of this week’s lab is to

<ul>

 <li>Implement grid search, gradient descent and stochastic gradient descent.</li>

 <li>Learn to debug your implementations.</li>

 <li>Learn to visualize results.</li>

 <li>Understand advantages and disadvantages of these algorithms.</li>

 <li>Study the effect of outliers using MSE and MAE cost functions.</li>

</ul>

Setup, data and sample code.               Obtain the folder labs/ex02 of the course github repository

<a href="https://github.com/epfml/ML_course/tree/master/labs/ex02">github.com/epfml/ML</a> <a href="https://github.com/epfml/ML_course/tree/master/labs/ex02">course</a>

We will use the dataset height weight genders.csv in this exercise, and we have provided sample code templates that already contain useful snippets of code required for this exercise.

You will be working in the notebook ex02.ipynb for all exercises of this week, by filling in the corresponding functions. The notebook already provides a lot of template code, as well as code to load the data, and normalize the features, visualize the results.

Additionally, please also take a look at the files helpers.py and plots.py, and make sure you understand them.

<h1>1           Computing the Cost Function</h1>

In this exercise, we will focus on simple linear regression which takes the following form,

<em>y<sub>n </sub></em>≈ <em>f</em>(<em>x<sub>n</sub></em><sub>1</sub>) = <em>w</em><sub>0 </sub>+ <em>w</em><sub>1</sub><em>x<sub>n</sub></em><sub>1</sub><em>.                                                                              </em>(1)

We will use height as the input variable <em>x<sub>n</sub></em><sub>1 </sub>and weight as the output variable <em>y<sub>n</sub></em>. The coefficients <em>w</em><sub>0 </sub>and <em>w</em><sub>1 </sub>are also called <em>model parameters</em>. We will use a mean-square-error (MSE) function defined as follows,

<em>.                                  </em>(2)

Our goal is to find and that minimize this <em>cost</em>.

Let us start by the array data type in <em>NumPy</em>. We store all the (<em>y<sub>n</sub>,x<sub>n</sub></em><sub>1</sub>) pairs in a vector and a matrix as shown below.

 <em>y</em><sub>1 </sub>                       1       <em>x</em>11 

 <em>y</em><sub>2 </sub>  1 <em>x</em><sub>21 </sub> <em>y </em>=  …  <em>X</em>f = <sub></sub> … … <sub></sub><sup></sup> (3)





<em>y</em><em>N                                              </em>1     <em>x<sub>N</sub></em>1

Exercise 1:

To understand this data format, answer the following warmup questions:

<ul>

 <li>What does each <em>column </em>of <em>X</em><sup>˜ </sup>represent?</li>

 <li>What does each <em>row </em>of <em>X</em><sup>˜ </sup>represent?</li>

 <li>Why do we have 1’s in <em>X</em><sup>˜ </sup>?</li>

 <li>If we have heights and weights of 3 people, what would be the size of <em>y </em>and <em>X</em><sup>˜ </sup>? What would <em>X</em><sup>˜ </sup><sub>32 </sub>represent?</li>

 <li>In helpers.py, we have already provided code to form arrays for <em>y </em>and <em>X</em><sup>˜ </sup>. Have a look at the code, and make sure you understand how they are constructed.</li>

 <li>Check if the sizes of the variables make sense (use functions shape).</li>

</ul>

<ol>

 <li>a) Now we will compute the MSE. Let us introduce the vector notation <em>e </em>= <em>y </em>− <em>Xw</em><sup>˜ </sup>, for given model parameters <em>w </em>= [<em>w</em><sub>0</sub><em>, w</em><sub>1</sub>]<sup>&gt;</sup>. Prove that the MSE can also be rewritten in terms of the vector <em>e</em>, as</li>

</ol>

L(<em>w</em>) = <em>…            </em>(4) b) Complete the implementation of the notebook function compute loss(y, tx, w). You can start by setting <em>w </em>= [1<em>,</em>2]<sup>&gt;</sup>, and test your function.

<h1>2           Grid Search</h1>

Now we are ready to implement our first optimization algorithm: Grid Search. Revise the lecture notes.

Exercise 2:

<ol>

 <li>Fill in the notebook function grid search(y, tx, w0, w1) to implement grid search. You will have to write one for-loop per dimension, and compute the cost function for each setting of <em>w</em><sub>0 </sub>and <em>w</em><sub>1</sub>. Once you have all values of cost function stored in the variable loss, the code finds an approximate minimum (as discussed in the class).</li>

</ol>

The code should print the obtained minimum value of the cost function along with the found and. It should also show a contour plot and the plot of the fit, as shown in Figure 1.

Figure 1: Grid Search Visualization

<ol>

 <li>Does this look like a good estimate? Why not? What is the problem? Why is the MSE plot not smooth?</li>

</ol>

Repeat the above exercise by changing the grid spacing to 10 instead of 50. Compare the new fit to the old one.

<ol>

 <li>Discuss with your peers:

  <ul>

   <li>To obtain an accurate fit, do you need a coarse grid or a fine grid?</li>

   <li>Try different values of grid spacing. What do you observe?</li>

   <li>How does increasing the number of values affect the computational cost? How fast or slow does your code run?</li>

  </ul></li>

</ol>

<h1>3           Gradient Descent</h1>

In the lecture, we derived the following expressions for the gradient (the vector of partial derivatives) of the MSE for linear regression,

(5)

(6)

Denoting the gradient by ∇L(<em>w</em>), we can write these operations in vector form as follows,

<em>X</em><sup>˜ </sup><sup>&gt;</sup><em>e                               </em>(7)

Exercise 3:

<ol>

 <li>Now implement a function that computes the gradients. Implement the notebook function compute gradient(y, tx, w) using Equation (7). Verify that the function returns the right values. First, manually compute the gradients for hand-picked values of <em>y</em>, <em>X</em><sup>˜ </sup>, and <em>w </em>and compare them to the output of compute gradient.</li>

 <li>Once you make sure that your gradient code is correct, get some intuition about the gradient values:</li>

</ol>

Compute the gradients for

<ul>

 <li><em>w</em><sub>0 </sub>= 100 and <em>w</em><sub>1 </sub>= 20</li>

 <li><em>w</em><sub>0 </sub>= 50 and <em>w</em><sub>1 </sub>= 10</li>

</ul>

What do the values of these gradients tell us? For example, think about the norm of this vector. In which case are they bigger? What does that mean?

<em>Hint: </em>Imagine a quadratic function and estimate its gradient near its minimum and far from it. <em>Hint 2: </em>As we know from the lecture notes, the update rule for gradient descent at step <em>t </em>is

<em>w</em>(<em>t</em>+1) = <em>w</em>(<em>t</em>) − <em>γ </em>∇L(<em>w</em>(<em>t</em>))                                                                        (8)

where <em>γ &gt; </em>0 is the step size, and ∇L ∈ R<sup>2 </sup>is the gradient vector.

<ol>

 <li>Fill in the notebook function gradient descent(y, tx, initial w, …). Run the code and visualize the iterations. Also, look at the printed messages that show L and values of and. Take a detailed look at these plots,

  <ul>

   <li>Is the cost being minimized?</li>

   <li>Is the algorithm converging? What can be said about the convergence speed?</li>

   <li>How good are the final values of <em>w</em><sub>1 </sub>and <em>w</em><sub>0 </sub>found?</li>

  </ul></li>

 <li>Now let’s experiment with the value of the step size and initialization parameters and see how they influences the convergence. In theory, gradient descent converges to the optimum on convex functions, when the value of the step size is chosen appropriately.

  <ul>

   <li>Try the following values of step size: 0.001, 0.01, 0.5, 1, 2, 2.5. What do you observe? Did the procedure converge?</li>

   <li>Try different initializations with fixed step size <em>γ </em>= 0<em>.</em>1, for instance:

    <ul>

     <li><em>w</em><sub>0 </sub>= 0, <em>w</em><sub>1 </sub>= 0</li>

     <li><em>w</em><sub>0 </sub>= 100, <em>w</em><sub>1 </sub>= 10</li>

     <li><em>w</em><sub>0 </sub>= −1000, <em>w</em><sub>1 </sub>= 1000</li>

    </ul></li>

  </ul></li>

</ol>

What do you observe? Did the procedure converge?

<h1>4           Stochastic Gradient Descent</h1>

Exercise 4:

Let us implement stochastic gradient descent. Recall from the lecture notes that the update rule for stochastic

gradient descent on an objective function  at step <em>t </em>is

<em>w</em>(<em>t</em>+1) = <em>w</em>(<em>t</em>) − <em>γ </em>∇L<em>n</em>(<em>w</em>(<em>t</em>)) <em>.                                                                           </em>(9)

HINT: You can use the function batch iter() in the file of helpers.py to generate mini-batch data for stochastic gradient descent.

<h1>5           Effect of Outliers and MAE Cost Function</h1>

In the course we talked about <em>outliers</em>. Outliers might occur due to measurement errors. For example, in the weight/height data, a coding mistake could introduce points whose weight is measured in pounds rather than kilograms.

Such outlier points may have a strong influence on model parameters. For example, MSE (the one you implemented above) is known to be sensitive to outliers, as discussed in the class.

Exercise 5:

Let’s simulate the presence of two outliers, and their effect on linear regression under MSE cost function,

<ul>

 <li>Reload the data through function load data() by setting sub sample=True to keep only a few data examples.</li>

 <li>Plot the data. You should get a cloud of points similar, but less dense, than what you saw before with the whole dataset.</li>

 <li>As before, find the values of <em>w</em><sub>0</sub><em>,w</em><sub>1 </sub>to fit a linear model (using MSE cost function), and plot the resulting <em>f </em>together with the data points.</li>

 <li>Now we will add two outliers points simulating the mistake that we entered the weights in pounds instead of kilograms. For example, you can achieve this by setting add outlier=True in load data(). Feel free to add more outlier points.</li>

 <li>Fit the model again to the augmented dataset with the outliers. Does it look like a good fit?</li>

</ul>

One way to deal with outliers is to use a more <em>robust </em>cost function, such as the Mean Absolute Error (MAE), as discussed in the class.

<h1>6           Subgradient Descent</h1>

Exercise 6:

Modify the function compute loss(y, tx, w) for the Mean Absolute Error cost function.

Unfortunately, you cannot directly use gradient descent, since the MAE function is non-differentiable at several points.

<ol>

 <li>Compute a subgradient of the MAE cost function, for every given vector <em>w</em>.</li>

</ol>

<em>Hint: Use the chain rule to compute the subgradient of the absolute value function. For a function </em>L(<em>w</em>) := <em>h</em>(<em>q</em>(<em>w</em>)) <em>with </em><em>q differentiable, the subgradient can be computed using </em><em>∂</em>L(<em>w</em>) = <em>∂h</em>(<em>q</em>(<em>w</em>))·∇<em>q</em>(<em>w</em>)<em>, where each </em><em>∂.. denotes the set of all subgradient vectors.</em>

<ol>

 <li>Implement subgradient descent for the MAE cost function.</li>

</ol>

To do so, write a new function compute gradient(y, tx, w) for the new MAE objective, and modify it to return a subgradient if the given <em>w </em>turns out to be a non-differentiable point.

Plot the resulting model <em>f </em>together with the two curves obtained in the previous exercise.

<ul>

 <li>Is the fit using MAE <em>better </em>than the one using MSE?</li>

 <li>Did your optimization algorithm ever encounter a non-differentiable point?</li>

</ul>

<ol>

 <li>Implement stochastic subgradient descent (SGD) for the MAE cost function.</li>

</ol>

How is the picture different when you compare the two algorithm variants on MAE, compared to what you have observed on MSE?