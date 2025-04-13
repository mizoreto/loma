def sum_array(arr : In[Array[float]], arr_size : In[int]) -> float:     
    # The "In" of In[int] means you can't modify it; if it's "Out", it means you can't read from it. 
    # See ../../tests/loma_code/call_stmt.py for the example of using "Out"
    i : int = 0
    s : float = 0.0
    while (i < arr_size, max_iter := 1000):
        s = s + arr[i]
        i = i + 1
    s_relu : float = 0.0
    if s > 0:
    	s_relu = s
    return s_relu
