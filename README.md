# CartPole-v1 Advantage Actor Critic (A2C) in Keras

CartPole-v1 is an environment presented by OpenAI Gym. In this repository we have implemeted Advantage Actor Critic (A2C) algorithm in Keras for building an agent to solve CartPole-v1 environment.

### Commands to run
 * **To train the model**
  
        python train_model.py
        
 *  **To test the model**

        python test_model.py 'path_of_saved_model_weights' (without quotes)

 * **To test agent with our trained weights**
        
        python test_model.py saved_model/500.0.h5


### Results

 *  **Output of agent taking random actions**
 
      ![Episode: 0](demo/cartpole_v1_random.gif)

 * **Output of our agent at Episode: 85 with score 500.0**
        
      ![Episode: 85, Score:500.0](demo/cartpole_v1_a2c_ours.gif)



