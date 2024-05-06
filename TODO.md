<h1>TODO</h1>
<ol>
    <li>Instead of randomly splitting the dataset into training and test datasets, make sure all data from the same hurricane stays in either training or test dataset.</li>
    <li>Incorporate the cnn data (or the vectorization of the image data) into the model. We can try the following two methods. I honestly think that neither of these would work and think that the second one is going perform horrible but it is an easy way to show prof that we have tried something.</li>
    <ul>
        <li>Just include the vectorized image data as an input at the first timestep. The next timestep will try to predict the vector at the next timestep but we will not directly apply loss function on the vectorized image data. Instead, we will apply the loss function on the predicted values of the other features.</li>
        <li>We will also apply the loss function on the predicted vectorized image data</li>
    </ul>
    <li>Incorporate pressure data (have not been able to find a good dataset). Also if found, think about what chunk of the atmospheric pressure data we are going to plug in. One thing we could do is to train a model to predict the atmospheric pressures of the near locations shown in the picture.</li>
</ol>