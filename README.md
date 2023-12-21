To run the code, just run the main.py file with an input argument of dataset's
file path, for example: wifi_db/noisy_dataset.txt, it will do everything and 
generate a visualization of the result matrix of cross validation test.

If you want to run the visulization of the tree  trained on the entire clean dataset
, uncomment the two lines
    # root, depth = decision_tree_learning(x, y, 0)
    # create_plot(root, depth)

and comment out the other lines below them