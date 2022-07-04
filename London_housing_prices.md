# Springboard Data Science Career Track Unit 4 Challenge - Tier 3 Complete

## Objectives
Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time. 

In this notebook, you're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the *slightly* messier work that data scientists do with actual datasets!

Here’s the mystery we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***


A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London [(here's some info for the curious)](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.

***This is the Tier 3 notebook, which means it's not filled in at all: we'll just give you the skeleton of a project, the brief and the data. It's up to you to play around with it and see what you can find out! Good luck! If you struggle, feel free to look at easier tiers for help; but try to dip in and out of them, as the more independent work you do, the better it is for your learning!***

This challenge will make use of only what you learned in the following DataCamp courses: 
- Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
- Data Types for Data Science
- Python Data Science Toolbox (Part One) 
- pandas Foundations
- Manipulating DataFrames with pandas
- Merging DataFrames with pandas

Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following: 
- **pandas**
    - **data ingestion and inspection** (pandas Foundations, Module One) 
    - **exploratory data analysis** (pandas Foundations, Module Two)
    - **tidying and cleaning** (Manipulating DataFrames with pandas, Module Three) 
    - **transforming DataFrames** (Manipulating DataFrames with pandas, Module One)
    - **subsetting DataFrames with lists** (Manipulating DataFrames with pandas, Module One) 
    - **filtering DataFrames** (Manipulating DataFrames with pandas, Module One) 
    - **grouping data** (Manipulating DataFrames with pandas, Module Four) 
    - **melting data** (Manipulating DataFrames with pandas, Module Three) 
    - **advanced indexing** (Manipulating DataFrames with pandas, Module Four) 
- **matplotlib** (Intermediate Python for Data Science, Module One)
- **fundamental data types** (Data Types for Data Science, Module One) 
- **dictionaries** (Intermediate Python for Data Science, Module Two)
- **handling dates and times** (Data Types for Data Science, Module Four)
- **function definition** (Python Data Science Toolbox - Part One, Module One)
- **default arguments, variable length, and scope** (Python Data Science Toolbox - Part One, Module Two) 
- **lambda functions and error handling** (Python Data Science Toolbox - Part One, Module Four) 

## The Data Science Pipeline

This is Tier Three, so we'll get you started. But after that, it's all in your hands! When you feel done with your investigations, look back over what you've accomplished, and prepare a quick presentation of your findings for the next mentor meeting. 

Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as  [David Spiegelhalter](https://www.youtube.com/watch?v=oUs1uvsz0Ok) reminds us, there is no substitute for simply **taking a really, really good look at the data.** Sometimes, this is all we need to answer our question.

Data Science projects generally adhere to the four stages of Data Science Pipeline:
1. Sourcing and loading 
2. Cleaning, transforming, and visualizing 
3. Modeling 
4. Evaluating and concluding 


### 1. Sourcing and Loading 

Any Data Science project kicks off by importing  ***pandas***. The documentation of this wonderful library can be found [here](https://pandas.pydata.org/). As you've seen, pandas is conveniently connected to the [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) libraries. 

***Hint:*** This part of the data science pipeline will test those skills you acquired in the pandas Foundations course, Module One. 

#### 1.1. Importing Libraries


```python
# Let's import the pandas, numpy libraries as pd, and np respectively. 
import pandas as pd
import numpy as np

# Load the pyplot collection of functions from matplotlib, as plt 
import matplotlib.pyplot as plt
```

#### 1.2.  Loading the data
Your data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal for London-oriented datasets. 


```python
# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices = "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)
```

### 2. Cleaning, transforming, and visualizing
This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.

The end goal of data cleaning is to have tidy data. When data is tidy: 

1. Each variable has a column.
2. Each observation forms a row.

Keep the end goal in mind as you move through this process, every step will take you closer. 



***Hint:*** This part of the data science pipeline should test those skills you acquired in: 
- Intermediate Python for data science, all modules.
- pandas Foundations, all modules. 
- Manipulating DataFrames with pandas, all modules.
- Data Types for Data Science, Module Four.
- Python Data Science Toolbox - Part One, all modules

**2.1. Exploring your data** 

Think about your pandas functions for checking out a dataframe. 


```python
properties.shape

properties.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>...</th>
      <th>NORTH WEST</th>
      <th>YORKS &amp; THE HUMBER</th>
      <th>EAST MIDLANDS</th>
      <th>WEST MIDLANDS</th>
      <th>EAST OF ENGLAND</th>
      <th>LONDON</th>
      <th>SOUTH EAST</th>
      <th>SOUTH WEST</th>
      <th>Unnamed: 47</th>
      <th>England</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaT</td>
      <td>E09000001</td>
      <td>E09000002</td>
      <td>E09000003</td>
      <td>E09000004</td>
      <td>E09000005</td>
      <td>E09000006</td>
      <td>E09000007</td>
      <td>E09000008</td>
      <td>E09000009</td>
      <td>...</td>
      <td>E12000002</td>
      <td>E12000003</td>
      <td>E12000004</td>
      <td>E12000005</td>
      <td>E12000006</td>
      <td>E12000007</td>
      <td>E12000008</td>
      <td>E12000009</td>
      <td>NaN</td>
      <td>E92000001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1995-01-01</td>
      <td>91448.98487</td>
      <td>50460.2266</td>
      <td>93284.51832</td>
      <td>64958.09036</td>
      <td>71306.56698</td>
      <td>81671.47692</td>
      <td>120932.8881</td>
      <td>69158.16225</td>
      <td>79885.89069</td>
      <td>...</td>
      <td>43958.48001</td>
      <td>44803.42878</td>
      <td>45544.52227</td>
      <td>48527.52339</td>
      <td>56701.5961</td>
      <td>74435.76052</td>
      <td>64018.87894</td>
      <td>54705.1579</td>
      <td>NaN</td>
      <td>53202.77128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1995-02-01</td>
      <td>82202.77314</td>
      <td>51085.77983</td>
      <td>93190.16963</td>
      <td>64787.92069</td>
      <td>72022.26197</td>
      <td>81657.55944</td>
      <td>119508.8622</td>
      <td>68951.09542</td>
      <td>80897.06551</td>
      <td>...</td>
      <td>43925.42289</td>
      <td>44528.80721</td>
      <td>46051.57066</td>
      <td>49341.29029</td>
      <td>56593.59475</td>
      <td>72777.93709</td>
      <td>63715.02399</td>
      <td>54356.14843</td>
      <td>NaN</td>
      <td>53096.1549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995-03-01</td>
      <td>79120.70256</td>
      <td>51268.96956</td>
      <td>92247.52435</td>
      <td>64367.49344</td>
      <td>72015.76274</td>
      <td>81449.31143</td>
      <td>120282.2131</td>
      <td>68712.44341</td>
      <td>81379.86288</td>
      <td>...</td>
      <td>44434.8681</td>
      <td>45200.46775</td>
      <td>45383.82395</td>
      <td>49442.17973</td>
      <td>56171.18278</td>
      <td>73896.84204</td>
      <td>64113.60858</td>
      <td>53583.07667</td>
      <td>NaN</td>
      <td>53201.2843</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995-04-01</td>
      <td>77101.20804</td>
      <td>53133.50526</td>
      <td>90762.87492</td>
      <td>64277.66881</td>
      <td>72965.63094</td>
      <td>81124.41227</td>
      <td>120097.899</td>
      <td>68610.04641</td>
      <td>82188.90498</td>
      <td>...</td>
      <td>44267.7796</td>
      <td>45614.34341</td>
      <td>46124.23045</td>
      <td>49455.93299</td>
      <td>56567.89582</td>
      <td>74455.28754</td>
      <td>64623.22395</td>
      <td>54786.01938</td>
      <td>NaN</td>
      <td>53590.8548</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



**2.2. Cleaning the data**

You might find you need to transpose your dataframe, check out what its row indexes are, and reset the index. You  also might find you need to assign the values of the first row to your column headings  . (Hint: recall the .columns feature of DataFrames, as well as the iloc[] method).

Don't be afraid to use StackOverflow for help  with this.


```python
properties_transpose = properties.T
properties_transpose
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>319</th>
      <th>320</th>
      <th>321</th>
      <th>322</th>
      <th>323</th>
      <th>324</th>
      <th>325</th>
      <th>326</th>
      <th>327</th>
      <th>328</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>1995-09-01 00:00:00</td>
      <td>...</td>
      <td>2021-07-01 00:00:00</td>
      <td>2021-08-01 00:00:00</td>
      <td>2021-09-01 00:00:00</td>
      <td>2021-10-01 00:00:00</td>
      <td>2021-11-01 00:00:00</td>
      <td>2021-12-01 00:00:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>2022-02-01 00:00:00</td>
      <td>2022-03-01 00:00:00</td>
      <td>2022-04-01 00:00:00</td>
    </tr>
    <tr>
      <th>City of London</th>
      <td>E09000001</td>
      <td>91448.98487</td>
      <td>82202.77314</td>
      <td>79120.70256</td>
      <td>77101.20804</td>
      <td>84409.14932</td>
      <td>94900.51244</td>
      <td>110128.0423</td>
      <td>112329.4376</td>
      <td>104473.1096</td>
      <td>...</td>
      <td>946823.5048</td>
      <td>951748.3655</td>
      <td>956411.5828</td>
      <td>792257.5493</td>
      <td>798212.3879</td>
      <td>813435.102</td>
      <td>811508.8041</td>
      <td>864897.234</td>
      <td>813511.6177</td>
      <td>838145.2394</td>
    </tr>
    <tr>
      <th>Barking &amp; Dagenham</th>
      <td>E09000002</td>
      <td>50460.2266</td>
      <td>51085.77983</td>
      <td>51268.96956</td>
      <td>53133.50526</td>
      <td>53042.24852</td>
      <td>53700.34831</td>
      <td>52113.12157</td>
      <td>52232.19868</td>
      <td>51471.61353</td>
      <td>...</td>
      <td>309272.9988</td>
      <td>313458.6524</td>
      <td>315061.8135</td>
      <td>326733.2244</td>
      <td>327928.3074</td>
      <td>329557.0047</td>
      <td>338869.7029</td>
      <td>345001.3576</td>
      <td>346064.5259</td>
      <td>336125.8982</td>
    </tr>
    <tr>
      <th>Barnet</th>
      <td>E09000003</td>
      <td>93284.51832</td>
      <td>93190.16963</td>
      <td>92247.52435</td>
      <td>90762.87492</td>
      <td>90258.00033</td>
      <td>90107.23471</td>
      <td>91441.24768</td>
      <td>92361.31512</td>
      <td>93273.12245</td>
      <td>...</td>
      <td>543469.3212</td>
      <td>557782.6161</td>
      <td>556321.1527</td>
      <td>574852.5269</td>
      <td>577708.2605</td>
      <td>583195.3087</td>
      <td>580179.6373</td>
      <td>576485.8755</td>
      <td>576689.7495</td>
      <td>596898.1846</td>
    </tr>
    <tr>
      <th>Bexley</th>
      <td>E09000004</td>
      <td>64958.09036</td>
      <td>64787.92069</td>
      <td>64367.49344</td>
      <td>64277.66881</td>
      <td>63997.13588</td>
      <td>64252.32335</td>
      <td>63722.70055</td>
      <td>64432.60005</td>
      <td>64509.54767</td>
      <td>...</td>
      <td>366095.067</td>
      <td>371295.1503</td>
      <td>372283.9658</td>
      <td>375783.3263</td>
      <td>378961.3767</td>
      <td>379526.3458</td>
      <td>382270.4215</td>
      <td>387171.1018</td>
      <td>393246.1533</td>
      <td>392154.9821</td>
    </tr>
    <tr>
      <th>Brent</th>
      <td>E09000005</td>
      <td>71306.56698</td>
      <td>72022.26197</td>
      <td>72015.76274</td>
      <td>72965.63094</td>
      <td>73704.04743</td>
      <td>74310.48167</td>
      <td>74127.03788</td>
      <td>73547.0411</td>
      <td>73789.54287</td>
      <td>...</td>
      <td>525953.4981</td>
      <td>519967.3789</td>
      <td>516871.8266</td>
      <td>509397.4875</td>
      <td>515067.9083</td>
      <td>518958.5856</td>
      <td>519652.186</td>
      <td>525664.3639</td>
      <td>532006.6781</td>
      <td>546934.1163</td>
    </tr>
    <tr>
      <th>Bromley</th>
      <td>E09000006</td>
      <td>81671.47692</td>
      <td>81657.55944</td>
      <td>81449.31143</td>
      <td>81124.41227</td>
      <td>81542.61561</td>
      <td>82382.83435</td>
      <td>82898.52264</td>
      <td>82054.37156</td>
      <td>81440.43008</td>
      <td>...</td>
      <td>470181.4754</td>
      <td>471749.7903</td>
      <td>472042.6102</td>
      <td>474662.6913</td>
      <td>478724.0107</td>
      <td>476969.7846</td>
      <td>477940.5654</td>
      <td>482488.4382</td>
      <td>488167.0019</td>
      <td>495549.8738</td>
    </tr>
    <tr>
      <th>Camden</th>
      <td>E09000007</td>
      <td>120932.8881</td>
      <td>119508.8622</td>
      <td>120282.2131</td>
      <td>120097.899</td>
      <td>119929.2782</td>
      <td>121887.4625</td>
      <td>124027.5768</td>
      <td>125529.8039</td>
      <td>120596.8511</td>
      <td>...</td>
      <td>870227.6164</td>
      <td>872206.3792</td>
      <td>871771.2508</td>
      <td>838987.1191</td>
      <td>868463.619</td>
      <td>889160.0227</td>
      <td>862346.1416</td>
      <td>844906.0505</td>
      <td>860277.973</td>
      <td>874080.5433</td>
    </tr>
    <tr>
      <th>Croydon</th>
      <td>E09000008</td>
      <td>69158.16225</td>
      <td>68951.09542</td>
      <td>68712.44341</td>
      <td>68610.04641</td>
      <td>68844.9169</td>
      <td>69052.51103</td>
      <td>69142.48112</td>
      <td>68993.42545</td>
      <td>69393.50023</td>
      <td>...</td>
      <td>392224.4554</td>
      <td>395412.777</td>
      <td>396949.9723</td>
      <td>402995.4493</td>
      <td>403922.9444</td>
      <td>404118.1402</td>
      <td>402782.4332</td>
      <td>406092.0046</td>
      <td>412616.1388</td>
      <td>417946.8683</td>
    </tr>
    <tr>
      <th>Ealing</th>
      <td>E09000009</td>
      <td>79885.89069</td>
      <td>80897.06551</td>
      <td>81379.86288</td>
      <td>82188.90498</td>
      <td>82077.05525</td>
      <td>81630.66181</td>
      <td>82352.2226</td>
      <td>82706.65927</td>
      <td>82011.08271</td>
      <td>...</td>
      <td>505598.6796</td>
      <td>513230.0546</td>
      <td>511840.9236</td>
      <td>518710.3974</td>
      <td>518947.3247</td>
      <td>516851.7744</td>
      <td>518836.7152</td>
      <td>525004.1651</td>
      <td>532883.1655</td>
      <td>525438.3054</td>
    </tr>
    <tr>
      <th>Enfield</th>
      <td>E09000010</td>
      <td>72514.69096</td>
      <td>73155.19746</td>
      <td>72190.44144</td>
      <td>71442.92235</td>
      <td>70630.77955</td>
      <td>71348.31147</td>
      <td>71837.54011</td>
      <td>72237.94562</td>
      <td>71725.22104</td>
      <td>...</td>
      <td>419798.2409</td>
      <td>426530.5084</td>
      <td>428512.4521</td>
      <td>433383.3177</td>
      <td>433406.7133</td>
      <td>432808.7906</td>
      <td>430460.8348</td>
      <td>433629.3853</td>
      <td>435443.7257</td>
      <td>449967.405</td>
    </tr>
    <tr>
      <th>Greenwich</th>
      <td>E09000011</td>
      <td>62300.10169</td>
      <td>60993.26863</td>
      <td>61377.83464</td>
      <td>61927.7246</td>
      <td>63512.99103</td>
      <td>64751.56404</td>
      <td>65486.34112</td>
      <td>65076.43195</td>
      <td>63996.81525</td>
      <td>...</td>
      <td>395899.9541</td>
      <td>402903.2678</td>
      <td>398582.6741</td>
      <td>412563.2251</td>
      <td>410415.2112</td>
      <td>408466.531</td>
      <td>416151.3221</td>
      <td>423822.6677</td>
      <td>430997.3337</td>
      <td>434433.3002</td>
    </tr>
    <tr>
      <th>Hackney</th>
      <td>E09000012</td>
      <td>61296.52637</td>
      <td>63187.08332</td>
      <td>63593.29935</td>
      <td>65139.64403</td>
      <td>66193.99212</td>
      <td>66921.17101</td>
      <td>68390.753</td>
      <td>68096.79385</td>
      <td>68752.50284</td>
      <td>...</td>
      <td>614581.973</td>
      <td>623178.8426</td>
      <td>617334.4059</td>
      <td>590645.1197</td>
      <td>593564.6053</td>
      <td>604999.7028</td>
      <td>631181.1859</td>
      <td>635288.5923</td>
      <td>623277.649</td>
      <td>607030.0269</td>
    </tr>
    <tr>
      <th>Hammersmith &amp; Fulham</th>
      <td>E09000013</td>
      <td>124902.8602</td>
      <td>122087.718</td>
      <td>120635.9467</td>
      <td>121424.6241</td>
      <td>124433.539</td>
      <td>126175.1513</td>
      <td>124381.5134</td>
      <td>123625.3196</td>
      <td>123094.0484</td>
      <td>...</td>
      <td>792899.0146</td>
      <td>842383.1755</td>
      <td>855023.4364</td>
      <td>800617.9054</td>
      <td>770570.3519</td>
      <td>770921.4663</td>
      <td>757865.9621</td>
      <td>759519.0329</td>
      <td>744871.9549</td>
      <td>721184.8016</td>
    </tr>
    <tr>
      <th>Haringey</th>
      <td>E09000014</td>
      <td>76287.56947</td>
      <td>78901.21036</td>
      <td>78521.94855</td>
      <td>79545.57477</td>
      <td>79374.0349</td>
      <td>79956.3621</td>
      <td>80746.34881</td>
      <td>81217.69074</td>
      <td>82142.89052</td>
      <td>...</td>
      <td>558903.1279</td>
      <td>576359.0143</td>
      <td>580884.9813</td>
      <td>594907.8638</td>
      <td>587721.1266</td>
      <td>579905.8303</td>
      <td>585812.3312</td>
      <td>588085.1792</td>
      <td>591510.3968</td>
      <td>601439.0498</td>
    </tr>
    <tr>
      <th>Harrow</th>
      <td>E09000015</td>
      <td>84769.52599</td>
      <td>83396.10525</td>
      <td>83416.23759</td>
      <td>83567.88439</td>
      <td>83853.65615</td>
      <td>84173.24689</td>
      <td>84226.69844</td>
      <td>84430.61796</td>
      <td>83606.97863</td>
      <td>...</td>
      <td>482448.7115</td>
      <td>483634.2004</td>
      <td>489234.959</td>
      <td>490181.5935</td>
      <td>493970.9287</td>
      <td>500948.9798</td>
      <td>510587.5528</td>
      <td>521087.5114</td>
      <td>513438.9373</td>
      <td>497236.0168</td>
    </tr>
    <tr>
      <th>Havering</th>
      <td>E09000016</td>
      <td>68000.13774</td>
      <td>69393.51294</td>
      <td>69368.02407</td>
      <td>69444.26215</td>
      <td>68534.52248</td>
      <td>68464.60664</td>
      <td>68680.83996</td>
      <td>69023.36482</td>
      <td>68108.186</td>
      <td>...</td>
      <td>387971.7013</td>
      <td>391094.531</td>
      <td>397761.0702</td>
      <td>399522.2364</td>
      <td>402636.1549</td>
      <td>402159.1905</td>
      <td>406196.2793</td>
      <td>409503.7945</td>
      <td>411629.0926</td>
      <td>425898.3484</td>
    </tr>
    <tr>
      <th>Hillingdon</th>
      <td>E09000017</td>
      <td>73834.82964</td>
      <td>75031.0696</td>
      <td>74188.66949</td>
      <td>73911.40591</td>
      <td>73117.12416</td>
      <td>74005.00585</td>
      <td>74671.13263</td>
      <td>74967.86534</td>
      <td>73843.55239</td>
      <td>...</td>
      <td>428077.3619</td>
      <td>432756.7161</td>
      <td>435584.3392</td>
      <td>439270.7253</td>
      <td>442996.8034</td>
      <td>441142.9155</td>
      <td>443353.1842</td>
      <td>446984.0262</td>
      <td>453411.8981</td>
      <td>459564.5274</td>
    </tr>
    <tr>
      <th>Hounslow</th>
      <td>E09000018</td>
      <td>72231.70537</td>
      <td>71051.55852</td>
      <td>72097.99411</td>
      <td>71890.28339</td>
      <td>72877.47219</td>
      <td>72331.08116</td>
      <td>73717.78844</td>
      <td>74479.94802</td>
      <td>74426.6609</td>
      <td>...</td>
      <td>413375.8441</td>
      <td>421713.5103</td>
      <td>420017.2445</td>
      <td>427966.1737</td>
      <td>427796.9489</td>
      <td>430102.3059</td>
      <td>429894.727</td>
      <td>434880.6259</td>
      <td>426963.7612</td>
      <td>428482.634</td>
    </tr>
    <tr>
      <th>Islington</th>
      <td>E09000019</td>
      <td>92516.48557</td>
      <td>94342.37334</td>
      <td>93465.86407</td>
      <td>93344.49305</td>
      <td>94346.39917</td>
      <td>97428.94311</td>
      <td>98976.14077</td>
      <td>98951.20791</td>
      <td>99582.63778</td>
      <td>...</td>
      <td>681337.1403</td>
      <td>701323.6472</td>
      <td>705058.7877</td>
      <td>715697.3299</td>
      <td>717451.5942</td>
      <td>727162.7293</td>
      <td>709242.6839</td>
      <td>701379.8325</td>
      <td>712058.7368</td>
      <td>712968.1253</td>
    </tr>
    <tr>
      <th>Kensington &amp; Chelsea</th>
      <td>E09000020</td>
      <td>182694.8326</td>
      <td>182345.2463</td>
      <td>182878.8231</td>
      <td>184176.9168</td>
      <td>191474.1141</td>
      <td>197265.7602</td>
      <td>197963.3169</td>
      <td>198037.4218</td>
      <td>197047.8333</td>
      <td>...</td>
      <td>1330982.998</td>
      <td>1401419.923</td>
      <td>1454839.208</td>
      <td>1458613.808</td>
      <td>1438989.829</td>
      <td>1398510.565</td>
      <td>1352740.932</td>
      <td>1367973.262</td>
      <td>1416959.804</td>
      <td>1513711.487</td>
    </tr>
    <tr>
      <th>Kingston upon Thames</th>
      <td>E09000021</td>
      <td>80875.84843</td>
      <td>81230.13524</td>
      <td>81111.48848</td>
      <td>81672.80476</td>
      <td>82123.51084</td>
      <td>82205.66822</td>
      <td>82525.793</td>
      <td>83342.84552</td>
      <td>85110.96835</td>
      <td>...</td>
      <td>505183.5533</td>
      <td>514134.2926</td>
      <td>516015.3129</td>
      <td>531847.0545</td>
      <td>534055.568</td>
      <td>529322.5767</td>
      <td>533147.9659</td>
      <td>528461.6059</td>
      <td>530418.5572</td>
      <td>528432.8212</td>
    </tr>
    <tr>
      <th>Lambeth</th>
      <td>E09000022</td>
      <td>67770.98843</td>
      <td>65381.51908</td>
      <td>66336.51868</td>
      <td>66388.7716</td>
      <td>69035.11076</td>
      <td>68881.15764</td>
      <td>69608.72242</td>
      <td>68840.02827</td>
      <td>70155.81997</td>
      <td>...</td>
      <td>523487.3447</td>
      <td>520885.169</td>
      <td>525268.1135</td>
      <td>540679.2602</td>
      <td>544539.4907</td>
      <td>538235.9944</td>
      <td>530907.0118</td>
      <td>539015.9961</td>
      <td>537596.0607</td>
      <td>545755.8001</td>
    </tr>
    <tr>
      <th>Lewisham</th>
      <td>E09000023</td>
      <td>60491.26109</td>
      <td>60869.27091</td>
      <td>60288.03002</td>
      <td>59471.03136</td>
      <td>58551.38387</td>
      <td>58041.43543</td>
      <td>58126.37811</td>
      <td>58151.3154</td>
      <td>58742.99034</td>
      <td>...</td>
      <td>423888.2707</td>
      <td>430117.7098</td>
      <td>428779.9942</td>
      <td>443617.909</td>
      <td>445022.3566</td>
      <td>441630.527</td>
      <td>439795.9001</td>
      <td>445844.9622</td>
      <td>458488.1819</td>
      <td>446546.4882</td>
    </tr>
    <tr>
      <th>Merton</th>
      <td>E09000024</td>
      <td>82070.6133</td>
      <td>79982.74872</td>
      <td>80661.68279</td>
      <td>79990.54333</td>
      <td>80873.98643</td>
      <td>80704.92667</td>
      <td>81055.90335</td>
      <td>80781.09186</td>
      <td>80824.70075</td>
      <td>...</td>
      <td>535944.3993</td>
      <td>543733.332</td>
      <td>546785.2293</td>
      <td>569622.5158</td>
      <td>575971.4498</td>
      <td>581821.3433</td>
      <td>567340.2989</td>
      <td>560501.3722</td>
      <td>556016.3395</td>
      <td>566919.3326</td>
    </tr>
    <tr>
      <th>Newham</th>
      <td>E09000025</td>
      <td>53539.31919</td>
      <td>53153.88306</td>
      <td>53458.26393</td>
      <td>54479.75395</td>
      <td>55803.95958</td>
      <td>56067.76986</td>
      <td>55458.31693</td>
      <td>54709.35467</td>
      <td>54585.50364</td>
      <td>...</td>
      <td>386305.0525</td>
      <td>390221.7543</td>
      <td>387319.1586</td>
      <td>386603.7566</td>
      <td>393351.5181</td>
      <td>392779.3325</td>
      <td>402920.9823</td>
      <td>400987.3735</td>
      <td>410900.7381</td>
      <td>407474.353</td>
    </tr>
    <tr>
      <th>Redbridge</th>
      <td>E09000026</td>
      <td>72189.58437</td>
      <td>72141.6261</td>
      <td>72501.35502</td>
      <td>72228.60295</td>
      <td>72366.64122</td>
      <td>72279.4325</td>
      <td>72880.83974</td>
      <td>73275.16891</td>
      <td>73204.02373</td>
      <td>...</td>
      <td>452175.7917</td>
      <td>456982.6321</td>
      <td>457727.2249</td>
      <td>467328.8624</td>
      <td>470083.2799</td>
      <td>472423.2905</td>
      <td>475348.0472</td>
      <td>475400.2126</td>
      <td>477530.9219</td>
      <td>480317.7664</td>
    </tr>
    <tr>
      <th>Richmond upon Thames</th>
      <td>E09000027</td>
      <td>109326.1245</td>
      <td>111103.0394</td>
      <td>107325.4742</td>
      <td>106875</td>
      <td>107707.6799</td>
      <td>112865.0542</td>
      <td>114656.6011</td>
      <td>112320.4096</td>
      <td>110424.5703</td>
      <td>...</td>
      <td>719965.9182</td>
      <td>737839.7697</td>
      <td>738180.4787</td>
      <td>742823.242</td>
      <td>736705.4341</td>
      <td>749842.4435</td>
      <td>746049.445</td>
      <td>759033.6523</td>
      <td>767844.9027</td>
      <td>792537.7549</td>
    </tr>
    <tr>
      <th>Southwark</th>
      <td>E09000028</td>
      <td>67885.20344</td>
      <td>64799.0648</td>
      <td>65763.29719</td>
      <td>63073.62117</td>
      <td>64420.49933</td>
      <td>64155.81449</td>
      <td>67024.74767</td>
      <td>65525.94434</td>
      <td>63467.00948</td>
      <td>...</td>
      <td>505876.4932</td>
      <td>504491.1796</td>
      <td>502749.5663</td>
      <td>502174.1096</td>
      <td>509073.2419</td>
      <td>515861.6355</td>
      <td>525810.0621</td>
      <td>532735.7856</td>
      <td>524979.9563</td>
      <td>527636.1363</td>
    </tr>
    <tr>
      <th>Sutton</th>
      <td>E09000029</td>
      <td>71536.97357</td>
      <td>70893.20851</td>
      <td>70306.83844</td>
      <td>69411.9439</td>
      <td>69759.21989</td>
      <td>70125.24728</td>
      <td>70789.57284</td>
      <td>69958.41918</td>
      <td>69937.40214</td>
      <td>...</td>
      <td>403539.1482</td>
      <td>408625.3226</td>
      <td>413668.9106</td>
      <td>414279.999</td>
      <td>410830.2969</td>
      <td>410539.7667</td>
      <td>417675.1441</td>
      <td>425631.3353</td>
      <td>426301.5912</td>
      <td>424726.2324</td>
    </tr>
    <tr>
      <th>Tower Hamlets</th>
      <td>E09000030</td>
      <td>59865.18995</td>
      <td>62318.53353</td>
      <td>63938.67686</td>
      <td>66233.19383</td>
      <td>66432.85846</td>
      <td>66232.16372</td>
      <td>64692.22672</td>
      <td>63472.27558</td>
      <td>62177.59435</td>
      <td>...</td>
      <td>452108.6757</td>
      <td>457922.4263</td>
      <td>452656.9125</td>
      <td>440824.7709</td>
      <td>439926.5357</td>
      <td>441675.8554</td>
      <td>457024.7988</td>
      <td>473180.4346</td>
      <td>478492.9401</td>
      <td>499715.8655</td>
    </tr>
    <tr>
      <th>Waltham Forest</th>
      <td>E09000031</td>
      <td>61319.44913</td>
      <td>60252.12246</td>
      <td>60871.08493</td>
      <td>60971.39722</td>
      <td>61494.16938</td>
      <td>61547.79643</td>
      <td>61933.52738</td>
      <td>61916.4222</td>
      <td>61548.15206</td>
      <td>...</td>
      <td>474663.277</td>
      <td>479318.4486</td>
      <td>481730.3344</td>
      <td>478621.8206</td>
      <td>475867.7176</td>
      <td>472600.6936</td>
      <td>480620.2518</td>
      <td>486643.1803</td>
      <td>491765.0393</td>
      <td>478260.832</td>
    </tr>
    <tr>
      <th>Wandsworth</th>
      <td>E09000032</td>
      <td>88559.04381</td>
      <td>88641.01678</td>
      <td>87124.81523</td>
      <td>87026.00225</td>
      <td>86518.05945</td>
      <td>88114.3351</td>
      <td>89830.58934</td>
      <td>90560.68078</td>
      <td>91373.6984</td>
      <td>...</td>
      <td>602572.5952</td>
      <td>621428.0568</td>
      <td>628957.2158</td>
      <td>639899.1276</td>
      <td>633861.1258</td>
      <td>634167.3798</td>
      <td>622914.2685</td>
      <td>628818.0618</td>
      <td>634288.9237</td>
      <td>641131.9851</td>
    </tr>
    <tr>
      <th>Westminster</th>
      <td>E09000033</td>
      <td>133025.2772</td>
      <td>131468.3096</td>
      <td>132260.3417</td>
      <td>133370.2036</td>
      <td>133911.1117</td>
      <td>134562.1941</td>
      <td>133450.2162</td>
      <td>136581.5082</td>
      <td>135993.0705</td>
      <td>...</td>
      <td>922437.8627</td>
      <td>1037515.224</td>
      <td>1035239.214</td>
      <td>1023849.29</td>
      <td>976110.3458</td>
      <td>1033172.155</td>
      <td>1014964.961</td>
      <td>1012521.105</td>
      <td>981804.544</td>
      <td>977553.758</td>
    </tr>
    <tr>
      <th>Unnamed: 34</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Inner London</th>
      <td>E13000001</td>
      <td>78251.9765</td>
      <td>75885.70201</td>
      <td>76591.59947</td>
      <td>76851.56697</td>
      <td>79129.19443</td>
      <td>79969.1525</td>
      <td>80550.47935</td>
      <td>80597.64563</td>
      <td>80402.85</td>
      <td>...</td>
      <td>591071.3615</td>
      <td>606309.7563</td>
      <td>608352.9803</td>
      <td>606922.2298</td>
      <td>605308.5658</td>
      <td>608152.9046</td>
      <td>607502.4373</td>
      <td>612361.1838</td>
      <td>614996.0728</td>
      <td>618426.0265</td>
    </tr>
    <tr>
      <th>Outer London</th>
      <td>E13000002</td>
      <td>72958.79836</td>
      <td>72937.88262</td>
      <td>72714.53478</td>
      <td>72591.92469</td>
      <td>72752.99414</td>
      <td>73189.39978</td>
      <td>73665.90517</td>
      <td>73691.12888</td>
      <td>73454.44292</td>
      <td>...</td>
      <td>451232.0296</td>
      <td>456747.401</td>
      <td>458059.3546</td>
      <td>464437.3659</td>
      <td>465894.515</td>
      <td>466650.5425</td>
      <td>468733.5111</td>
      <td>472821.2171</td>
      <td>476130.6331</td>
      <td>480501.3827</td>
    </tr>
    <tr>
      <th>Unnamed: 37</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>NORTH EAST</th>
      <td>E12000001</td>
      <td>42076.35411</td>
      <td>42571.98949</td>
      <td>42369.72984</td>
      <td>42095.8436</td>
      <td>43266.45165</td>
      <td>42315.34372</td>
      <td>43287.74323</td>
      <td>41899.05494</td>
      <td>41850.85646</td>
      <td>...</td>
      <td>141930.3081</td>
      <td>144243.9476</td>
      <td>155441.0771</td>
      <td>143930.1289</td>
      <td>147244.4946</td>
      <td>149189.3107</td>
      <td>150733.8356</td>
      <td>151948.0782</td>
      <td>155000.0783</td>
      <td>155214.9625</td>
    </tr>
    <tr>
      <th>NORTH WEST</th>
      <td>E12000002</td>
      <td>43958.48001</td>
      <td>43925.42289</td>
      <td>44434.8681</td>
      <td>44267.7796</td>
      <td>44223.61973</td>
      <td>44112.96432</td>
      <td>44109.58764</td>
      <td>44193.66583</td>
      <td>44088.07696</td>
      <td>...</td>
      <td>184281.5266</td>
      <td>190330.6752</td>
      <td>201604.0817</td>
      <td>189969.1405</td>
      <td>195790.068</td>
      <td>199043.1651</td>
      <td>200830.9409</td>
      <td>203694.787</td>
      <td>204719.0271</td>
      <td>208867.0279</td>
    </tr>
    <tr>
      <th>YORKS &amp; THE HUMBER</th>
      <td>E12000003</td>
      <td>44803.42878</td>
      <td>44528.80721</td>
      <td>45200.46775</td>
      <td>45614.34341</td>
      <td>44830.98563</td>
      <td>45392.63981</td>
      <td>45534.99864</td>
      <td>45111.45939</td>
      <td>44837.86023</td>
      <td>...</td>
      <td>180749.7466</td>
      <td>186824.8086</td>
      <td>196609.6373</td>
      <td>186675.4517</td>
      <td>191813.3804</td>
      <td>193706.5518</td>
      <td>195162.0767</td>
      <td>198675.8981</td>
      <td>198609.5008</td>
      <td>201805.9692</td>
    </tr>
    <tr>
      <th>EAST MIDLANDS</th>
      <td>E12000004</td>
      <td>45544.52227</td>
      <td>46051.57066</td>
      <td>45383.82395</td>
      <td>46124.23045</td>
      <td>45878.00396</td>
      <td>45679.99539</td>
      <td>46037.67312</td>
      <td>45922.53585</td>
      <td>45771.66321</td>
      <td>...</td>
      <td>212957.267</td>
      <td>219094.8058</td>
      <td>228456.9779</td>
      <td>224070.7577</td>
      <td>228070.0028</td>
      <td>230924.2495</td>
      <td>232783.5029</td>
      <td>235637.6375</td>
      <td>239091.39</td>
      <td>237903.8155</td>
    </tr>
    <tr>
      <th>WEST MIDLANDS</th>
      <td>E12000005</td>
      <td>48527.52339</td>
      <td>49341.29029</td>
      <td>49442.17973</td>
      <td>49455.93299</td>
      <td>50369.66188</td>
      <td>50100.43023</td>
      <td>49860.00809</td>
      <td>49598.45969</td>
      <td>49319.69715</td>
      <td>...</td>
      <td>217499.1813</td>
      <td>224764.5526</td>
      <td>231698.6039</td>
      <td>225314.9881</td>
      <td>229901.1179</td>
      <td>234295.4347</td>
      <td>236304.7075</td>
      <td>238480.7707</td>
      <td>240866.4576</td>
      <td>242144.9283</td>
    </tr>
    <tr>
      <th>EAST OF ENGLAND</th>
      <td>E12000006</td>
      <td>56701.5961</td>
      <td>56593.59475</td>
      <td>56171.18278</td>
      <td>56567.89582</td>
      <td>56479.80183</td>
      <td>56288.94557</td>
      <td>57242.30186</td>
      <td>56732.40547</td>
      <td>56259.28635</td>
      <td>...</td>
      <td>310253.279</td>
      <td>320640.6378</td>
      <td>329694.9921</td>
      <td>326561.0196</td>
      <td>334640.4365</td>
      <td>335892.6578</td>
      <td>338384.171</td>
      <td>344434.416</td>
      <td>343492.4128</td>
      <td>344942.9554</td>
    </tr>
    <tr>
      <th>LONDON</th>
      <td>E12000007</td>
      <td>74435.76052</td>
      <td>72777.93709</td>
      <td>73896.84204</td>
      <td>74455.28754</td>
      <td>75432.02786</td>
      <td>75606.24501</td>
      <td>75984.24079</td>
      <td>75529.34488</td>
      <td>74940.80872</td>
      <td>...</td>
      <td>502252.0856</td>
      <td>512279.8349</td>
      <td>510750.8219</td>
      <td>513951.6446</td>
      <td>513687.3707</td>
      <td>515087.9239</td>
      <td>517529.8233</td>
      <td>526585.5925</td>
      <td>524406.3787</td>
      <td>529828.5981</td>
    </tr>
    <tr>
      <th>SOUTH EAST</th>
      <td>E12000008</td>
      <td>64018.87894</td>
      <td>63715.02399</td>
      <td>64113.60858</td>
      <td>64623.22395</td>
      <td>64530.36358</td>
      <td>65511.008</td>
      <td>65224.88465</td>
      <td>64851.60429</td>
      <td>64352.47119</td>
      <td>...</td>
      <td>346159.5032</td>
      <td>355557.9994</td>
      <td>368264.1108</td>
      <td>363284.9925</td>
      <td>369234.5412</td>
      <td>371268.2627</td>
      <td>376973.7347</td>
      <td>378505.6677</td>
      <td>383792.3209</td>
      <td>382790.785</td>
    </tr>
    <tr>
      <th>SOUTH WEST</th>
      <td>E12000009</td>
      <td>54705.1579</td>
      <td>54356.14843</td>
      <td>53583.07667</td>
      <td>54786.01938</td>
      <td>54698.83831</td>
      <td>54420.15939</td>
      <td>54265.86368</td>
      <td>54365.71495</td>
      <td>54243.98694</td>
      <td>...</td>
      <td>275206.0294</td>
      <td>287976.1043</td>
      <td>302301.1627</td>
      <td>295591.1568</td>
      <td>303610.1649</td>
      <td>307137.5123</td>
      <td>308308.3452</td>
      <td>312527.9644</td>
      <td>312678.6798</td>
      <td>318610.4376</td>
    </tr>
    <tr>
      <th>Unnamed: 47</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>England</th>
      <td>E92000001</td>
      <td>53202.77128</td>
      <td>53096.1549</td>
      <td>53201.2843</td>
      <td>53590.8548</td>
      <td>53678.24041</td>
      <td>53735.15475</td>
      <td>53900.60633</td>
      <td>53600.31975</td>
      <td>53309.2331</td>
      <td>...</td>
      <td>268980.2011</td>
      <td>277328.3215</td>
      <td>288215.782</td>
      <td>280457.4775</td>
      <td>286268.2167</td>
      <td>289147.5126</td>
      <td>291644.4072</td>
      <td>295296.2716</td>
      <td>296945.9156</td>
      <td>299249.031</td>
    </tr>
  </tbody>
</table>
<p>49 rows × 329 columns</p>
</div>




```python
properties_transpose = properties_transpose.reset_index()
properties_transpose.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>319</th>
      <th>320</th>
      <th>321</th>
      <th>322</th>
      <th>323</th>
      <th>324</th>
      <th>325</th>
      <th>326</th>
      <th>327</th>
      <th>328</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2021-07-01 00:00:00</td>
      <td>2021-08-01 00:00:00</td>
      <td>2021-09-01 00:00:00</td>
      <td>2021-10-01 00:00:00</td>
      <td>2021-11-01 00:00:00</td>
      <td>2021-12-01 00:00:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>2022-02-01 00:00:00</td>
      <td>2022-03-01 00:00:00</td>
      <td>2022-04-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91448.98487</td>
      <td>82202.77314</td>
      <td>79120.70256</td>
      <td>77101.20804</td>
      <td>84409.14932</td>
      <td>94900.51244</td>
      <td>110128.0423</td>
      <td>112329.4376</td>
      <td>...</td>
      <td>946823.5048</td>
      <td>951748.3655</td>
      <td>956411.5828</td>
      <td>792257.5493</td>
      <td>798212.3879</td>
      <td>813435.102</td>
      <td>811508.8041</td>
      <td>864897.234</td>
      <td>813511.6177</td>
      <td>838145.2394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2266</td>
      <td>51085.77983</td>
      <td>51268.96956</td>
      <td>53133.50526</td>
      <td>53042.24852</td>
      <td>53700.34831</td>
      <td>52113.12157</td>
      <td>52232.19868</td>
      <td>...</td>
      <td>309272.9988</td>
      <td>313458.6524</td>
      <td>315061.8135</td>
      <td>326733.2244</td>
      <td>327928.3074</td>
      <td>329557.0047</td>
      <td>338869.7029</td>
      <td>345001.3576</td>
      <td>346064.5259</td>
      <td>336125.8982</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.51832</td>
      <td>93190.16963</td>
      <td>92247.52435</td>
      <td>90762.87492</td>
      <td>90258.00033</td>
      <td>90107.23471</td>
      <td>91441.24768</td>
      <td>92361.31512</td>
      <td>...</td>
      <td>543469.3212</td>
      <td>557782.6161</td>
      <td>556321.1527</td>
      <td>574852.5269</td>
      <td>577708.2605</td>
      <td>583195.3087</td>
      <td>580179.6373</td>
      <td>576485.8755</td>
      <td>576689.7495</td>
      <td>596898.1846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.09036</td>
      <td>64787.92069</td>
      <td>64367.49344</td>
      <td>64277.66881</td>
      <td>63997.13588</td>
      <td>64252.32335</td>
      <td>63722.70055</td>
      <td>64432.60005</td>
      <td>...</td>
      <td>366095.067</td>
      <td>371295.1503</td>
      <td>372283.9658</td>
      <td>375783.3263</td>
      <td>378961.3767</td>
      <td>379526.3458</td>
      <td>382270.4215</td>
      <td>387171.1018</td>
      <td>393246.1533</td>
      <td>392154.9821</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 330 columns</p>
</div>




```python
properties_transpose.iloc[[0]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>NaN</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2021-07-01 00:00:00</th>
      <th>2021-08-01 00:00:00</th>
      <th>2021-09-01 00:00:00</th>
      <th>2021-10-01 00:00:00</th>
      <th>2021-11-01 00:00:00</th>
      <th>2021-12-01 00:00:00</th>
      <th>2022-01-01 00:00:00</th>
      <th>2022-02-01 00:00:00</th>
      <th>2022-03-01 00:00:00</th>
      <th>2022-04-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2021-07-01 00:00:00</td>
      <td>2021-08-01 00:00:00</td>
      <td>2021-09-01 00:00:00</td>
      <td>2021-10-01 00:00:00</td>
      <td>2021-11-01 00:00:00</td>
      <td>2021-12-01 00:00:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>2022-02-01 00:00:00</td>
      <td>2022-03-01 00:00:00</td>
      <td>2022-04-01 00:00:00</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 330 columns</p>
</div>




```python
properties_transpose.columns = properties_transpose.iloc[0]
properties_transpose.columns
```




    Index([       'Unnamed: 0',                 NaT, 1995-01-01 00:00:00,
           1995-02-01 00:00:00, 1995-03-01 00:00:00, 1995-04-01 00:00:00,
           1995-05-01 00:00:00, 1995-06-01 00:00:00, 1995-07-01 00:00:00,
           1995-08-01 00:00:00,
           ...
           2021-07-01 00:00:00, 2021-08-01 00:00:00, 2021-09-01 00:00:00,
           2021-10-01 00:00:00, 2021-11-01 00:00:00, 2021-12-01 00:00:00,
           2022-01-01 00:00:00, 2022-02-01 00:00:00, 2022-03-01 00:00:00,
           2022-04-01 00:00:00],
          dtype='object', name=0, length=330)




```python
properties_transpose.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>NaN</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2021-07-01 00:00:00</th>
      <th>2021-08-01 00:00:00</th>
      <th>2021-09-01 00:00:00</th>
      <th>2021-10-01 00:00:00</th>
      <th>2021-11-01 00:00:00</th>
      <th>2021-12-01 00:00:00</th>
      <th>2022-01-01 00:00:00</th>
      <th>2022-02-01 00:00:00</th>
      <th>2022-03-01 00:00:00</th>
      <th>2022-04-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2021-07-01 00:00:00</td>
      <td>2021-08-01 00:00:00</td>
      <td>2021-09-01 00:00:00</td>
      <td>2021-10-01 00:00:00</td>
      <td>2021-11-01 00:00:00</td>
      <td>2021-12-01 00:00:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>2022-02-01 00:00:00</td>
      <td>2022-03-01 00:00:00</td>
      <td>2022-04-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91448.98487</td>
      <td>82202.77314</td>
      <td>79120.70256</td>
      <td>77101.20804</td>
      <td>84409.14932</td>
      <td>94900.51244</td>
      <td>110128.0423</td>
      <td>112329.4376</td>
      <td>...</td>
      <td>946823.5048</td>
      <td>951748.3655</td>
      <td>956411.5828</td>
      <td>792257.5493</td>
      <td>798212.3879</td>
      <td>813435.102</td>
      <td>811508.8041</td>
      <td>864897.234</td>
      <td>813511.6177</td>
      <td>838145.2394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2266</td>
      <td>51085.77983</td>
      <td>51268.96956</td>
      <td>53133.50526</td>
      <td>53042.24852</td>
      <td>53700.34831</td>
      <td>52113.12157</td>
      <td>52232.19868</td>
      <td>...</td>
      <td>309272.9988</td>
      <td>313458.6524</td>
      <td>315061.8135</td>
      <td>326733.2244</td>
      <td>327928.3074</td>
      <td>329557.0047</td>
      <td>338869.7029</td>
      <td>345001.3576</td>
      <td>346064.5259</td>
      <td>336125.8982</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.51832</td>
      <td>93190.16963</td>
      <td>92247.52435</td>
      <td>90762.87492</td>
      <td>90258.00033</td>
      <td>90107.23471</td>
      <td>91441.24768</td>
      <td>92361.31512</td>
      <td>...</td>
      <td>543469.3212</td>
      <td>557782.6161</td>
      <td>556321.1527</td>
      <td>574852.5269</td>
      <td>577708.2605</td>
      <td>583195.3087</td>
      <td>580179.6373</td>
      <td>576485.8755</td>
      <td>576689.7495</td>
      <td>596898.1846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.09036</td>
      <td>64787.92069</td>
      <td>64367.49344</td>
      <td>64277.66881</td>
      <td>63997.13588</td>
      <td>64252.32335</td>
      <td>63722.70055</td>
      <td>64432.60005</td>
      <td>...</td>
      <td>366095.067</td>
      <td>371295.1503</td>
      <td>372283.9658</td>
      <td>375783.3263</td>
      <td>378961.3767</td>
      <td>379526.3458</td>
      <td>382270.4215</td>
      <td>387171.1018</td>
      <td>393246.1533</td>
      <td>392154.9821</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 330 columns</p>
</div>




```python
properties_transpose.drop(0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>NaN</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2021-07-01 00:00:00</th>
      <th>2021-08-01 00:00:00</th>
      <th>2021-09-01 00:00:00</th>
      <th>2021-10-01 00:00:00</th>
      <th>2021-11-01 00:00:00</th>
      <th>2021-12-01 00:00:00</th>
      <th>2022-01-01 00:00:00</th>
      <th>2022-02-01 00:00:00</th>
      <th>2022-03-01 00:00:00</th>
      <th>2022-04-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91448.98487</td>
      <td>82202.77314</td>
      <td>79120.70256</td>
      <td>77101.20804</td>
      <td>84409.14932</td>
      <td>94900.51244</td>
      <td>110128.0423</td>
      <td>112329.4376</td>
      <td>...</td>
      <td>946823.5048</td>
      <td>951748.3655</td>
      <td>956411.5828</td>
      <td>792257.5493</td>
      <td>798212.3879</td>
      <td>813435.102</td>
      <td>811508.8041</td>
      <td>864897.234</td>
      <td>813511.6177</td>
      <td>838145.2394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2266</td>
      <td>51085.77983</td>
      <td>51268.96956</td>
      <td>53133.50526</td>
      <td>53042.24852</td>
      <td>53700.34831</td>
      <td>52113.12157</td>
      <td>52232.19868</td>
      <td>...</td>
      <td>309272.9988</td>
      <td>313458.6524</td>
      <td>315061.8135</td>
      <td>326733.2244</td>
      <td>327928.3074</td>
      <td>329557.0047</td>
      <td>338869.7029</td>
      <td>345001.3576</td>
      <td>346064.5259</td>
      <td>336125.8982</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.51832</td>
      <td>93190.16963</td>
      <td>92247.52435</td>
      <td>90762.87492</td>
      <td>90258.00033</td>
      <td>90107.23471</td>
      <td>91441.24768</td>
      <td>92361.31512</td>
      <td>...</td>
      <td>543469.3212</td>
      <td>557782.6161</td>
      <td>556321.1527</td>
      <td>574852.5269</td>
      <td>577708.2605</td>
      <td>583195.3087</td>
      <td>580179.6373</td>
      <td>576485.8755</td>
      <td>576689.7495</td>
      <td>596898.1846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.09036</td>
      <td>64787.92069</td>
      <td>64367.49344</td>
      <td>64277.66881</td>
      <td>63997.13588</td>
      <td>64252.32335</td>
      <td>63722.70055</td>
      <td>64432.60005</td>
      <td>...</td>
      <td>366095.067</td>
      <td>371295.1503</td>
      <td>372283.9658</td>
      <td>375783.3263</td>
      <td>378961.3767</td>
      <td>379526.3458</td>
      <td>382270.4215</td>
      <td>387171.1018</td>
      <td>393246.1533</td>
      <td>392154.9821</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>71306.56698</td>
      <td>72022.26197</td>
      <td>72015.76274</td>
      <td>72965.63094</td>
      <td>73704.04743</td>
      <td>74310.48167</td>
      <td>74127.03788</td>
      <td>73547.0411</td>
      <td>...</td>
      <td>525953.4981</td>
      <td>519967.3789</td>
      <td>516871.8266</td>
      <td>509397.4875</td>
      <td>515067.9083</td>
      <td>518958.5856</td>
      <td>519652.186</td>
      <td>525664.3639</td>
      <td>532006.6781</td>
      <td>546934.1163</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bromley</td>
      <td>E09000006</td>
      <td>81671.47692</td>
      <td>81657.55944</td>
      <td>81449.31143</td>
      <td>81124.41227</td>
      <td>81542.61561</td>
      <td>82382.83435</td>
      <td>82898.52264</td>
      <td>82054.37156</td>
      <td>...</td>
      <td>470181.4754</td>
      <td>471749.7903</td>
      <td>472042.6102</td>
      <td>474662.6913</td>
      <td>478724.0107</td>
      <td>476969.7846</td>
      <td>477940.5654</td>
      <td>482488.4382</td>
      <td>488167.0019</td>
      <td>495549.8738</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Camden</td>
      <td>E09000007</td>
      <td>120932.8881</td>
      <td>119508.8622</td>
      <td>120282.2131</td>
      <td>120097.899</td>
      <td>119929.2782</td>
      <td>121887.4625</td>
      <td>124027.5768</td>
      <td>125529.8039</td>
      <td>...</td>
      <td>870227.6164</td>
      <td>872206.3792</td>
      <td>871771.2508</td>
      <td>838987.1191</td>
      <td>868463.619</td>
      <td>889160.0227</td>
      <td>862346.1416</td>
      <td>844906.0505</td>
      <td>860277.973</td>
      <td>874080.5433</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Croydon</td>
      <td>E09000008</td>
      <td>69158.16225</td>
      <td>68951.09542</td>
      <td>68712.44341</td>
      <td>68610.04641</td>
      <td>68844.9169</td>
      <td>69052.51103</td>
      <td>69142.48112</td>
      <td>68993.42545</td>
      <td>...</td>
      <td>392224.4554</td>
      <td>395412.777</td>
      <td>396949.9723</td>
      <td>402995.4493</td>
      <td>403922.9444</td>
      <td>404118.1402</td>
      <td>402782.4332</td>
      <td>406092.0046</td>
      <td>412616.1388</td>
      <td>417946.8683</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ealing</td>
      <td>E09000009</td>
      <td>79885.89069</td>
      <td>80897.06551</td>
      <td>81379.86288</td>
      <td>82188.90498</td>
      <td>82077.05525</td>
      <td>81630.66181</td>
      <td>82352.2226</td>
      <td>82706.65927</td>
      <td>...</td>
      <td>505598.6796</td>
      <td>513230.0546</td>
      <td>511840.9236</td>
      <td>518710.3974</td>
      <td>518947.3247</td>
      <td>516851.7744</td>
      <td>518836.7152</td>
      <td>525004.1651</td>
      <td>532883.1655</td>
      <td>525438.3054</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Enfield</td>
      <td>E09000010</td>
      <td>72514.69096</td>
      <td>73155.19746</td>
      <td>72190.44144</td>
      <td>71442.92235</td>
      <td>70630.77955</td>
      <td>71348.31147</td>
      <td>71837.54011</td>
      <td>72237.94562</td>
      <td>...</td>
      <td>419798.2409</td>
      <td>426530.5084</td>
      <td>428512.4521</td>
      <td>433383.3177</td>
      <td>433406.7133</td>
      <td>432808.7906</td>
      <td>430460.8348</td>
      <td>433629.3853</td>
      <td>435443.7257</td>
      <td>449967.405</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Greenwich</td>
      <td>E09000011</td>
      <td>62300.10169</td>
      <td>60993.26863</td>
      <td>61377.83464</td>
      <td>61927.7246</td>
      <td>63512.99103</td>
      <td>64751.56404</td>
      <td>65486.34112</td>
      <td>65076.43195</td>
      <td>...</td>
      <td>395899.9541</td>
      <td>402903.2678</td>
      <td>398582.6741</td>
      <td>412563.2251</td>
      <td>410415.2112</td>
      <td>408466.531</td>
      <td>416151.3221</td>
      <td>423822.6677</td>
      <td>430997.3337</td>
      <td>434433.3002</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hackney</td>
      <td>E09000012</td>
      <td>61296.52637</td>
      <td>63187.08332</td>
      <td>63593.29935</td>
      <td>65139.64403</td>
      <td>66193.99212</td>
      <td>66921.17101</td>
      <td>68390.753</td>
      <td>68096.79385</td>
      <td>...</td>
      <td>614581.973</td>
      <td>623178.8426</td>
      <td>617334.4059</td>
      <td>590645.1197</td>
      <td>593564.6053</td>
      <td>604999.7028</td>
      <td>631181.1859</td>
      <td>635288.5923</td>
      <td>623277.649</td>
      <td>607030.0269</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hammersmith &amp; Fulham</td>
      <td>E09000013</td>
      <td>124902.8602</td>
      <td>122087.718</td>
      <td>120635.9467</td>
      <td>121424.6241</td>
      <td>124433.539</td>
      <td>126175.1513</td>
      <td>124381.5134</td>
      <td>123625.3196</td>
      <td>...</td>
      <td>792899.0146</td>
      <td>842383.1755</td>
      <td>855023.4364</td>
      <td>800617.9054</td>
      <td>770570.3519</td>
      <td>770921.4663</td>
      <td>757865.9621</td>
      <td>759519.0329</td>
      <td>744871.9549</td>
      <td>721184.8016</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Haringey</td>
      <td>E09000014</td>
      <td>76287.56947</td>
      <td>78901.21036</td>
      <td>78521.94855</td>
      <td>79545.57477</td>
      <td>79374.0349</td>
      <td>79956.3621</td>
      <td>80746.34881</td>
      <td>81217.69074</td>
      <td>...</td>
      <td>558903.1279</td>
      <td>576359.0143</td>
      <td>580884.9813</td>
      <td>594907.8638</td>
      <td>587721.1266</td>
      <td>579905.8303</td>
      <td>585812.3312</td>
      <td>588085.1792</td>
      <td>591510.3968</td>
      <td>601439.0498</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Harrow</td>
      <td>E09000015</td>
      <td>84769.52599</td>
      <td>83396.10525</td>
      <td>83416.23759</td>
      <td>83567.88439</td>
      <td>83853.65615</td>
      <td>84173.24689</td>
      <td>84226.69844</td>
      <td>84430.61796</td>
      <td>...</td>
      <td>482448.7115</td>
      <td>483634.2004</td>
      <td>489234.959</td>
      <td>490181.5935</td>
      <td>493970.9287</td>
      <td>500948.9798</td>
      <td>510587.5528</td>
      <td>521087.5114</td>
      <td>513438.9373</td>
      <td>497236.0168</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Havering</td>
      <td>E09000016</td>
      <td>68000.13774</td>
      <td>69393.51294</td>
      <td>69368.02407</td>
      <td>69444.26215</td>
      <td>68534.52248</td>
      <td>68464.60664</td>
      <td>68680.83996</td>
      <td>69023.36482</td>
      <td>...</td>
      <td>387971.7013</td>
      <td>391094.531</td>
      <td>397761.0702</td>
      <td>399522.2364</td>
      <td>402636.1549</td>
      <td>402159.1905</td>
      <td>406196.2793</td>
      <td>409503.7945</td>
      <td>411629.0926</td>
      <td>425898.3484</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hillingdon</td>
      <td>E09000017</td>
      <td>73834.82964</td>
      <td>75031.0696</td>
      <td>74188.66949</td>
      <td>73911.40591</td>
      <td>73117.12416</td>
      <td>74005.00585</td>
      <td>74671.13263</td>
      <td>74967.86534</td>
      <td>...</td>
      <td>428077.3619</td>
      <td>432756.7161</td>
      <td>435584.3392</td>
      <td>439270.7253</td>
      <td>442996.8034</td>
      <td>441142.9155</td>
      <td>443353.1842</td>
      <td>446984.0262</td>
      <td>453411.8981</td>
      <td>459564.5274</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Hounslow</td>
      <td>E09000018</td>
      <td>72231.70537</td>
      <td>71051.55852</td>
      <td>72097.99411</td>
      <td>71890.28339</td>
      <td>72877.47219</td>
      <td>72331.08116</td>
      <td>73717.78844</td>
      <td>74479.94802</td>
      <td>...</td>
      <td>413375.8441</td>
      <td>421713.5103</td>
      <td>420017.2445</td>
      <td>427966.1737</td>
      <td>427796.9489</td>
      <td>430102.3059</td>
      <td>429894.727</td>
      <td>434880.6259</td>
      <td>426963.7612</td>
      <td>428482.634</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Islington</td>
      <td>E09000019</td>
      <td>92516.48557</td>
      <td>94342.37334</td>
      <td>93465.86407</td>
      <td>93344.49305</td>
      <td>94346.39917</td>
      <td>97428.94311</td>
      <td>98976.14077</td>
      <td>98951.20791</td>
      <td>...</td>
      <td>681337.1403</td>
      <td>701323.6472</td>
      <td>705058.7877</td>
      <td>715697.3299</td>
      <td>717451.5942</td>
      <td>727162.7293</td>
      <td>709242.6839</td>
      <td>701379.8325</td>
      <td>712058.7368</td>
      <td>712968.1253</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kensington &amp; Chelsea</td>
      <td>E09000020</td>
      <td>182694.8326</td>
      <td>182345.2463</td>
      <td>182878.8231</td>
      <td>184176.9168</td>
      <td>191474.1141</td>
      <td>197265.7602</td>
      <td>197963.3169</td>
      <td>198037.4218</td>
      <td>...</td>
      <td>1330982.998</td>
      <td>1401419.923</td>
      <td>1454839.208</td>
      <td>1458613.808</td>
      <td>1438989.829</td>
      <td>1398510.565</td>
      <td>1352740.932</td>
      <td>1367973.262</td>
      <td>1416959.804</td>
      <td>1513711.487</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Kingston upon Thames</td>
      <td>E09000021</td>
      <td>80875.84843</td>
      <td>81230.13524</td>
      <td>81111.48848</td>
      <td>81672.80476</td>
      <td>82123.51084</td>
      <td>82205.66822</td>
      <td>82525.793</td>
      <td>83342.84552</td>
      <td>...</td>
      <td>505183.5533</td>
      <td>514134.2926</td>
      <td>516015.3129</td>
      <td>531847.0545</td>
      <td>534055.568</td>
      <td>529322.5767</td>
      <td>533147.9659</td>
      <td>528461.6059</td>
      <td>530418.5572</td>
      <td>528432.8212</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lambeth</td>
      <td>E09000022</td>
      <td>67770.98843</td>
      <td>65381.51908</td>
      <td>66336.51868</td>
      <td>66388.7716</td>
      <td>69035.11076</td>
      <td>68881.15764</td>
      <td>69608.72242</td>
      <td>68840.02827</td>
      <td>...</td>
      <td>523487.3447</td>
      <td>520885.169</td>
      <td>525268.1135</td>
      <td>540679.2602</td>
      <td>544539.4907</td>
      <td>538235.9944</td>
      <td>530907.0118</td>
      <td>539015.9961</td>
      <td>537596.0607</td>
      <td>545755.8001</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Lewisham</td>
      <td>E09000023</td>
      <td>60491.26109</td>
      <td>60869.27091</td>
      <td>60288.03002</td>
      <td>59471.03136</td>
      <td>58551.38387</td>
      <td>58041.43543</td>
      <td>58126.37811</td>
      <td>58151.3154</td>
      <td>...</td>
      <td>423888.2707</td>
      <td>430117.7098</td>
      <td>428779.9942</td>
      <td>443617.909</td>
      <td>445022.3566</td>
      <td>441630.527</td>
      <td>439795.9001</td>
      <td>445844.9622</td>
      <td>458488.1819</td>
      <td>446546.4882</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Merton</td>
      <td>E09000024</td>
      <td>82070.6133</td>
      <td>79982.74872</td>
      <td>80661.68279</td>
      <td>79990.54333</td>
      <td>80873.98643</td>
      <td>80704.92667</td>
      <td>81055.90335</td>
      <td>80781.09186</td>
      <td>...</td>
      <td>535944.3993</td>
      <td>543733.332</td>
      <td>546785.2293</td>
      <td>569622.5158</td>
      <td>575971.4498</td>
      <td>581821.3433</td>
      <td>567340.2989</td>
      <td>560501.3722</td>
      <td>556016.3395</td>
      <td>566919.3326</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Newham</td>
      <td>E09000025</td>
      <td>53539.31919</td>
      <td>53153.88306</td>
      <td>53458.26393</td>
      <td>54479.75395</td>
      <td>55803.95958</td>
      <td>56067.76986</td>
      <td>55458.31693</td>
      <td>54709.35467</td>
      <td>...</td>
      <td>386305.0525</td>
      <td>390221.7543</td>
      <td>387319.1586</td>
      <td>386603.7566</td>
      <td>393351.5181</td>
      <td>392779.3325</td>
      <td>402920.9823</td>
      <td>400987.3735</td>
      <td>410900.7381</td>
      <td>407474.353</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Redbridge</td>
      <td>E09000026</td>
      <td>72189.58437</td>
      <td>72141.6261</td>
      <td>72501.35502</td>
      <td>72228.60295</td>
      <td>72366.64122</td>
      <td>72279.4325</td>
      <td>72880.83974</td>
      <td>73275.16891</td>
      <td>...</td>
      <td>452175.7917</td>
      <td>456982.6321</td>
      <td>457727.2249</td>
      <td>467328.8624</td>
      <td>470083.2799</td>
      <td>472423.2905</td>
      <td>475348.0472</td>
      <td>475400.2126</td>
      <td>477530.9219</td>
      <td>480317.7664</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Richmond upon Thames</td>
      <td>E09000027</td>
      <td>109326.1245</td>
      <td>111103.0394</td>
      <td>107325.4742</td>
      <td>106875</td>
      <td>107707.6799</td>
      <td>112865.0542</td>
      <td>114656.6011</td>
      <td>112320.4096</td>
      <td>...</td>
      <td>719965.9182</td>
      <td>737839.7697</td>
      <td>738180.4787</td>
      <td>742823.242</td>
      <td>736705.4341</td>
      <td>749842.4435</td>
      <td>746049.445</td>
      <td>759033.6523</td>
      <td>767844.9027</td>
      <td>792537.7549</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Southwark</td>
      <td>E09000028</td>
      <td>67885.20344</td>
      <td>64799.0648</td>
      <td>65763.29719</td>
      <td>63073.62117</td>
      <td>64420.49933</td>
      <td>64155.81449</td>
      <td>67024.74767</td>
      <td>65525.94434</td>
      <td>...</td>
      <td>505876.4932</td>
      <td>504491.1796</td>
      <td>502749.5663</td>
      <td>502174.1096</td>
      <td>509073.2419</td>
      <td>515861.6355</td>
      <td>525810.0621</td>
      <td>532735.7856</td>
      <td>524979.9563</td>
      <td>527636.1363</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Sutton</td>
      <td>E09000029</td>
      <td>71536.97357</td>
      <td>70893.20851</td>
      <td>70306.83844</td>
      <td>69411.9439</td>
      <td>69759.21989</td>
      <td>70125.24728</td>
      <td>70789.57284</td>
      <td>69958.41918</td>
      <td>...</td>
      <td>403539.1482</td>
      <td>408625.3226</td>
      <td>413668.9106</td>
      <td>414279.999</td>
      <td>410830.2969</td>
      <td>410539.7667</td>
      <td>417675.1441</td>
      <td>425631.3353</td>
      <td>426301.5912</td>
      <td>424726.2324</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Tower Hamlets</td>
      <td>E09000030</td>
      <td>59865.18995</td>
      <td>62318.53353</td>
      <td>63938.67686</td>
      <td>66233.19383</td>
      <td>66432.85846</td>
      <td>66232.16372</td>
      <td>64692.22672</td>
      <td>63472.27558</td>
      <td>...</td>
      <td>452108.6757</td>
      <td>457922.4263</td>
      <td>452656.9125</td>
      <td>440824.7709</td>
      <td>439926.5357</td>
      <td>441675.8554</td>
      <td>457024.7988</td>
      <td>473180.4346</td>
      <td>478492.9401</td>
      <td>499715.8655</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Waltham Forest</td>
      <td>E09000031</td>
      <td>61319.44913</td>
      <td>60252.12246</td>
      <td>60871.08493</td>
      <td>60971.39722</td>
      <td>61494.16938</td>
      <td>61547.79643</td>
      <td>61933.52738</td>
      <td>61916.4222</td>
      <td>...</td>
      <td>474663.277</td>
      <td>479318.4486</td>
      <td>481730.3344</td>
      <td>478621.8206</td>
      <td>475867.7176</td>
      <td>472600.6936</td>
      <td>480620.2518</td>
      <td>486643.1803</td>
      <td>491765.0393</td>
      <td>478260.832</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Wandsworth</td>
      <td>E09000032</td>
      <td>88559.04381</td>
      <td>88641.01678</td>
      <td>87124.81523</td>
      <td>87026.00225</td>
      <td>86518.05945</td>
      <td>88114.3351</td>
      <td>89830.58934</td>
      <td>90560.68078</td>
      <td>...</td>
      <td>602572.5952</td>
      <td>621428.0568</td>
      <td>628957.2158</td>
      <td>639899.1276</td>
      <td>633861.1258</td>
      <td>634167.3798</td>
      <td>622914.2685</td>
      <td>628818.0618</td>
      <td>634288.9237</td>
      <td>641131.9851</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Westminster</td>
      <td>E09000033</td>
      <td>133025.2772</td>
      <td>131468.3096</td>
      <td>132260.3417</td>
      <td>133370.2036</td>
      <td>133911.1117</td>
      <td>134562.1941</td>
      <td>133450.2162</td>
      <td>136581.5082</td>
      <td>...</td>
      <td>922437.8627</td>
      <td>1037515.224</td>
      <td>1035239.214</td>
      <td>1023849.29</td>
      <td>976110.3458</td>
      <td>1033172.155</td>
      <td>1014964.961</td>
      <td>1012521.105</td>
      <td>981804.544</td>
      <td>977553.758</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Inner London</td>
      <td>E13000001</td>
      <td>78251.9765</td>
      <td>75885.70201</td>
      <td>76591.59947</td>
      <td>76851.56697</td>
      <td>79129.19443</td>
      <td>79969.1525</td>
      <td>80550.47935</td>
      <td>80597.64563</td>
      <td>...</td>
      <td>591071.3615</td>
      <td>606309.7563</td>
      <td>608352.9803</td>
      <td>606922.2298</td>
      <td>605308.5658</td>
      <td>608152.9046</td>
      <td>607502.4373</td>
      <td>612361.1838</td>
      <td>614996.0728</td>
      <td>618426.0265</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Outer London</td>
      <td>E13000002</td>
      <td>72958.79836</td>
      <td>72937.88262</td>
      <td>72714.53478</td>
      <td>72591.92469</td>
      <td>72752.99414</td>
      <td>73189.39978</td>
      <td>73665.90517</td>
      <td>73691.12888</td>
      <td>...</td>
      <td>451232.0296</td>
      <td>456747.401</td>
      <td>458059.3546</td>
      <td>464437.3659</td>
      <td>465894.515</td>
      <td>466650.5425</td>
      <td>468733.5111</td>
      <td>472821.2171</td>
      <td>476130.6331</td>
      <td>480501.3827</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NORTH EAST</td>
      <td>E12000001</td>
      <td>42076.35411</td>
      <td>42571.98949</td>
      <td>42369.72984</td>
      <td>42095.8436</td>
      <td>43266.45165</td>
      <td>42315.34372</td>
      <td>43287.74323</td>
      <td>41899.05494</td>
      <td>...</td>
      <td>141930.3081</td>
      <td>144243.9476</td>
      <td>155441.0771</td>
      <td>143930.1289</td>
      <td>147244.4946</td>
      <td>149189.3107</td>
      <td>150733.8356</td>
      <td>151948.0782</td>
      <td>155000.0783</td>
      <td>155214.9625</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NORTH WEST</td>
      <td>E12000002</td>
      <td>43958.48001</td>
      <td>43925.42289</td>
      <td>44434.8681</td>
      <td>44267.7796</td>
      <td>44223.61973</td>
      <td>44112.96432</td>
      <td>44109.58764</td>
      <td>44193.66583</td>
      <td>...</td>
      <td>184281.5266</td>
      <td>190330.6752</td>
      <td>201604.0817</td>
      <td>189969.1405</td>
      <td>195790.068</td>
      <td>199043.1651</td>
      <td>200830.9409</td>
      <td>203694.787</td>
      <td>204719.0271</td>
      <td>208867.0279</td>
    </tr>
    <tr>
      <th>40</th>
      <td>YORKS &amp; THE HUMBER</td>
      <td>E12000003</td>
      <td>44803.42878</td>
      <td>44528.80721</td>
      <td>45200.46775</td>
      <td>45614.34341</td>
      <td>44830.98563</td>
      <td>45392.63981</td>
      <td>45534.99864</td>
      <td>45111.45939</td>
      <td>...</td>
      <td>180749.7466</td>
      <td>186824.8086</td>
      <td>196609.6373</td>
      <td>186675.4517</td>
      <td>191813.3804</td>
      <td>193706.5518</td>
      <td>195162.0767</td>
      <td>198675.8981</td>
      <td>198609.5008</td>
      <td>201805.9692</td>
    </tr>
    <tr>
      <th>41</th>
      <td>EAST MIDLANDS</td>
      <td>E12000004</td>
      <td>45544.52227</td>
      <td>46051.57066</td>
      <td>45383.82395</td>
      <td>46124.23045</td>
      <td>45878.00396</td>
      <td>45679.99539</td>
      <td>46037.67312</td>
      <td>45922.53585</td>
      <td>...</td>
      <td>212957.267</td>
      <td>219094.8058</td>
      <td>228456.9779</td>
      <td>224070.7577</td>
      <td>228070.0028</td>
      <td>230924.2495</td>
      <td>232783.5029</td>
      <td>235637.6375</td>
      <td>239091.39</td>
      <td>237903.8155</td>
    </tr>
    <tr>
      <th>42</th>
      <td>WEST MIDLANDS</td>
      <td>E12000005</td>
      <td>48527.52339</td>
      <td>49341.29029</td>
      <td>49442.17973</td>
      <td>49455.93299</td>
      <td>50369.66188</td>
      <td>50100.43023</td>
      <td>49860.00809</td>
      <td>49598.45969</td>
      <td>...</td>
      <td>217499.1813</td>
      <td>224764.5526</td>
      <td>231698.6039</td>
      <td>225314.9881</td>
      <td>229901.1179</td>
      <td>234295.4347</td>
      <td>236304.7075</td>
      <td>238480.7707</td>
      <td>240866.4576</td>
      <td>242144.9283</td>
    </tr>
    <tr>
      <th>43</th>
      <td>EAST OF ENGLAND</td>
      <td>E12000006</td>
      <td>56701.5961</td>
      <td>56593.59475</td>
      <td>56171.18278</td>
      <td>56567.89582</td>
      <td>56479.80183</td>
      <td>56288.94557</td>
      <td>57242.30186</td>
      <td>56732.40547</td>
      <td>...</td>
      <td>310253.279</td>
      <td>320640.6378</td>
      <td>329694.9921</td>
      <td>326561.0196</td>
      <td>334640.4365</td>
      <td>335892.6578</td>
      <td>338384.171</td>
      <td>344434.416</td>
      <td>343492.4128</td>
      <td>344942.9554</td>
    </tr>
    <tr>
      <th>44</th>
      <td>LONDON</td>
      <td>E12000007</td>
      <td>74435.76052</td>
      <td>72777.93709</td>
      <td>73896.84204</td>
      <td>74455.28754</td>
      <td>75432.02786</td>
      <td>75606.24501</td>
      <td>75984.24079</td>
      <td>75529.34488</td>
      <td>...</td>
      <td>502252.0856</td>
      <td>512279.8349</td>
      <td>510750.8219</td>
      <td>513951.6446</td>
      <td>513687.3707</td>
      <td>515087.9239</td>
      <td>517529.8233</td>
      <td>526585.5925</td>
      <td>524406.3787</td>
      <td>529828.5981</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SOUTH EAST</td>
      <td>E12000008</td>
      <td>64018.87894</td>
      <td>63715.02399</td>
      <td>64113.60858</td>
      <td>64623.22395</td>
      <td>64530.36358</td>
      <td>65511.008</td>
      <td>65224.88465</td>
      <td>64851.60429</td>
      <td>...</td>
      <td>346159.5032</td>
      <td>355557.9994</td>
      <td>368264.1108</td>
      <td>363284.9925</td>
      <td>369234.5412</td>
      <td>371268.2627</td>
      <td>376973.7347</td>
      <td>378505.6677</td>
      <td>383792.3209</td>
      <td>382790.785</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SOUTH WEST</td>
      <td>E12000009</td>
      <td>54705.1579</td>
      <td>54356.14843</td>
      <td>53583.07667</td>
      <td>54786.01938</td>
      <td>54698.83831</td>
      <td>54420.15939</td>
      <td>54265.86368</td>
      <td>54365.71495</td>
      <td>...</td>
      <td>275206.0294</td>
      <td>287976.1043</td>
      <td>302301.1627</td>
      <td>295591.1568</td>
      <td>303610.1649</td>
      <td>307137.5123</td>
      <td>308308.3452</td>
      <td>312527.9644</td>
      <td>312678.6798</td>
      <td>318610.4376</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>48</th>
      <td>England</td>
      <td>E92000001</td>
      <td>53202.77128</td>
      <td>53096.1549</td>
      <td>53201.2843</td>
      <td>53590.8548</td>
      <td>53678.24041</td>
      <td>53735.15475</td>
      <td>53900.60633</td>
      <td>53600.31975</td>
      <td>...</td>
      <td>268980.2011</td>
      <td>277328.3215</td>
      <td>288215.782</td>
      <td>280457.4775</td>
      <td>286268.2167</td>
      <td>289147.5126</td>
      <td>291644.4072</td>
      <td>295296.2716</td>
      <td>296945.9156</td>
      <td>299249.031</td>
    </tr>
  </tbody>
</table>
<p>48 rows × 330 columns</p>
</div>



**2.3. Cleaning the data (part 2)**

You might we have to **rename** a couple columns. How do you do this? The clue's pretty bold...


```python
properties_transpose = properties_transpose.rename(columns = {'Unnamed: 0':'London_Borough', pd.NaT: 'ID'})
properties_transpose.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>ID</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2021-07-01 00:00:00</th>
      <th>2021-08-01 00:00:00</th>
      <th>2021-09-01 00:00:00</th>
      <th>2021-10-01 00:00:00</th>
      <th>2021-11-01 00:00:00</th>
      <th>2021-12-01 00:00:00</th>
      <th>2022-01-01 00:00:00</th>
      <th>2022-02-01 00:00:00</th>
      <th>2022-03-01 00:00:00</th>
      <th>2022-04-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>NaT</td>
      <td>1995-01-01 00:00:00</td>
      <td>1995-02-01 00:00:00</td>
      <td>1995-03-01 00:00:00</td>
      <td>1995-04-01 00:00:00</td>
      <td>1995-05-01 00:00:00</td>
      <td>1995-06-01 00:00:00</td>
      <td>1995-07-01 00:00:00</td>
      <td>1995-08-01 00:00:00</td>
      <td>...</td>
      <td>2021-07-01 00:00:00</td>
      <td>2021-08-01 00:00:00</td>
      <td>2021-09-01 00:00:00</td>
      <td>2021-10-01 00:00:00</td>
      <td>2021-11-01 00:00:00</td>
      <td>2021-12-01 00:00:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>2022-02-01 00:00:00</td>
      <td>2022-03-01 00:00:00</td>
      <td>2022-04-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91448.98487</td>
      <td>82202.77314</td>
      <td>79120.70256</td>
      <td>77101.20804</td>
      <td>84409.14932</td>
      <td>94900.51244</td>
      <td>110128.0423</td>
      <td>112329.4376</td>
      <td>...</td>
      <td>946823.5048</td>
      <td>951748.3655</td>
      <td>956411.5828</td>
      <td>792257.5493</td>
      <td>798212.3879</td>
      <td>813435.102</td>
      <td>811508.8041</td>
      <td>864897.234</td>
      <td>813511.6177</td>
      <td>838145.2394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.2266</td>
      <td>51085.77983</td>
      <td>51268.96956</td>
      <td>53133.50526</td>
      <td>53042.24852</td>
      <td>53700.34831</td>
      <td>52113.12157</td>
      <td>52232.19868</td>
      <td>...</td>
      <td>309272.9988</td>
      <td>313458.6524</td>
      <td>315061.8135</td>
      <td>326733.2244</td>
      <td>327928.3074</td>
      <td>329557.0047</td>
      <td>338869.7029</td>
      <td>345001.3576</td>
      <td>346064.5259</td>
      <td>336125.8982</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.51832</td>
      <td>93190.16963</td>
      <td>92247.52435</td>
      <td>90762.87492</td>
      <td>90258.00033</td>
      <td>90107.23471</td>
      <td>91441.24768</td>
      <td>92361.31512</td>
      <td>...</td>
      <td>543469.3212</td>
      <td>557782.6161</td>
      <td>556321.1527</td>
      <td>574852.5269</td>
      <td>577708.2605</td>
      <td>583195.3087</td>
      <td>580179.6373</td>
      <td>576485.8755</td>
      <td>576689.7495</td>
      <td>596898.1846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.09036</td>
      <td>64787.92069</td>
      <td>64367.49344</td>
      <td>64277.66881</td>
      <td>63997.13588</td>
      <td>64252.32335</td>
      <td>63722.70055</td>
      <td>64432.60005</td>
      <td>...</td>
      <td>366095.067</td>
      <td>371295.1503</td>
      <td>372283.9658</td>
      <td>375783.3263</td>
      <td>378961.3767</td>
      <td>379526.3458</td>
      <td>382270.4215</td>
      <td>387171.1018</td>
      <td>393246.1533</td>
      <td>392154.9821</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 330 columns</p>
</div>




```python
properties_transpose = properties_transpose.drop(0)
```

**2.4.Transforming the data**

Remember what Wes McKinney said about tidy data? 

You might need to **melt** your DataFrame here. 


```python
properties_tidy = pd.melt(properties_transpose, id_vars= ['London_Borough', 'ID'])
properties_tidy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>ID</th>
      <th>0</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98487</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.2266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.51832</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09036</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.56698</td>
    </tr>
  </tbody>
</table>
</div>



Remember to make sure your column data types are all correct. Average prices, for example, should be floating point numbers... 


```python
properties_tidy = properties_tidy.rename(columns = {0 : "Date", "value":"Average Prices"} )
properties_tidy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>ID</th>
      <th>DATE</th>
      <th>Average Prices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98487</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.2266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.51832</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09036</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.56698</td>
    </tr>
  </tbody>
</table>
</div>




```python
properties_tidy.dtypes
```




    London_Borough            object
    ID                        object
    DATE              datetime64[ns]
    Average Prices            object
    dtype: object




```python
properties_tidy['Average Prices'] = pd.to_numeric(properties_tidy['Average Prices'])
properties_tidy.dtypes
```




    London_Borough            object
    ID                        object
    DATE              datetime64[ns]
    Average Prices           float64
    dtype: object



**2.5. Cleaning the data (part 3)**

Do we have an equal number of observations in the ID, Average Price, Month, and London Borough columns? Remember that there are only 32 London Boroughs. How many entries do you have in that column? 

Check out the contents of the London Borough column, and if you find null values, get rid of them however you see fit. 


```python
properties_tidy['London_Borough'].unique()
```




    array(['City of London', 'Barking & Dagenham', 'Barnet', 'Bexley',
           'Brent', 'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield',
           'Greenwich', 'Hackney', 'Hammersmith & Fulham', 'Haringey',
           'Harrow', 'Havering', 'Hillingdon', 'Hounslow', 'Islington',
           'Kensington & Chelsea', 'Kingston upon Thames', 'Lambeth',
           'Lewisham', 'Merton', 'Newham', 'Redbridge',
           'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets',
           'Waltham Forest', 'Wandsworth', 'Westminster', 'Unnamed: 34',
           'Inner London', 'Outer London', 'Unnamed: 37', 'NORTH EAST',
           'NORTH WEST', 'YORKS & THE HUMBER', 'EAST MIDLANDS',
           'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON', 'SOUTH EAST',
           'SOUTH WEST', 'Unnamed: 47', 'England'], dtype=object)




```python
properties_tidy[properties_tidy['ID'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>ID</th>
      <th>DATE</th>
      <th>Average Prices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>1995-01-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>1995-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>1995-02-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15684</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>2022-03-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15694</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>2022-03-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15729</th>
      <td>Unnamed: 34</td>
      <td>NaN</td>
      <td>2022-04-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15732</th>
      <td>Unnamed: 37</td>
      <td>NaN</td>
      <td>2022-04-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15742</th>
      <td>Unnamed: 47</td>
      <td>NaN</td>
      <td>2022-04-01</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>984 rows × 4 columns</p>
</div>




```python
DF_1 =  properties_tidy.dropna()
DF_1.head(50)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>ID</th>
      <th>DATE</th>
      <th>Average Prices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98487</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.22660</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.51832</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09036</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.56698</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bromley</td>
      <td>E09000006</td>
      <td>1995-01-01</td>
      <td>81671.47692</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Camden</td>
      <td>E09000007</td>
      <td>1995-01-01</td>
      <td>120932.88810</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croydon</td>
      <td>E09000008</td>
      <td>1995-01-01</td>
      <td>69158.16225</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ealing</td>
      <td>E09000009</td>
      <td>1995-01-01</td>
      <td>79885.89069</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Enfield</td>
      <td>E09000010</td>
      <td>1995-01-01</td>
      <td>72514.69096</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greenwich</td>
      <td>E09000011</td>
      <td>1995-01-01</td>
      <td>62300.10169</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hackney</td>
      <td>E09000012</td>
      <td>1995-01-01</td>
      <td>61296.52637</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hammersmith &amp; Fulham</td>
      <td>E09000013</td>
      <td>1995-01-01</td>
      <td>124902.86020</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Haringey</td>
      <td>E09000014</td>
      <td>1995-01-01</td>
      <td>76287.56947</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harrow</td>
      <td>E09000015</td>
      <td>1995-01-01</td>
      <td>84769.52599</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Havering</td>
      <td>E09000016</td>
      <td>1995-01-01</td>
      <td>68000.13774</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hillingdon</td>
      <td>E09000017</td>
      <td>1995-01-01</td>
      <td>73834.82964</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hounslow</td>
      <td>E09000018</td>
      <td>1995-01-01</td>
      <td>72231.70537</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Islington</td>
      <td>E09000019</td>
      <td>1995-01-01</td>
      <td>92516.48557</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kensington &amp; Chelsea</td>
      <td>E09000020</td>
      <td>1995-01-01</td>
      <td>182694.83260</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kingston upon Thames</td>
      <td>E09000021</td>
      <td>1995-01-01</td>
      <td>80875.84843</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lambeth</td>
      <td>E09000022</td>
      <td>1995-01-01</td>
      <td>67770.98843</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lewisham</td>
      <td>E09000023</td>
      <td>1995-01-01</td>
      <td>60491.26109</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Merton</td>
      <td>E09000024</td>
      <td>1995-01-01</td>
      <td>82070.61330</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Newham</td>
      <td>E09000025</td>
      <td>1995-01-01</td>
      <td>53539.31919</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Redbridge</td>
      <td>E09000026</td>
      <td>1995-01-01</td>
      <td>72189.58437</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Richmond upon Thames</td>
      <td>E09000027</td>
      <td>1995-01-01</td>
      <td>109326.12450</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Southwark</td>
      <td>E09000028</td>
      <td>1995-01-01</td>
      <td>67885.20344</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sutton</td>
      <td>E09000029</td>
      <td>1995-01-01</td>
      <td>71536.97357</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Tower Hamlets</td>
      <td>E09000030</td>
      <td>1995-01-01</td>
      <td>59865.18995</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Waltham Forest</td>
      <td>E09000031</td>
      <td>1995-01-01</td>
      <td>61319.44913</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Wandsworth</td>
      <td>E09000032</td>
      <td>1995-01-01</td>
      <td>88559.04381</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Westminster</td>
      <td>E09000033</td>
      <td>1995-01-01</td>
      <td>133025.27720</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Inner London</td>
      <td>E13000001</td>
      <td>1995-01-01</td>
      <td>78251.97650</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Outer London</td>
      <td>E13000002</td>
      <td>1995-01-01</td>
      <td>72958.79836</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NORTH EAST</td>
      <td>E12000001</td>
      <td>1995-01-01</td>
      <td>42076.35411</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NORTH WEST</td>
      <td>E12000002</td>
      <td>1995-01-01</td>
      <td>43958.48001</td>
    </tr>
    <tr>
      <th>39</th>
      <td>YORKS &amp; THE HUMBER</td>
      <td>E12000003</td>
      <td>1995-01-01</td>
      <td>44803.42878</td>
    </tr>
    <tr>
      <th>40</th>
      <td>EAST MIDLANDS</td>
      <td>E12000004</td>
      <td>1995-01-01</td>
      <td>45544.52227</td>
    </tr>
    <tr>
      <th>41</th>
      <td>WEST MIDLANDS</td>
      <td>E12000005</td>
      <td>1995-01-01</td>
      <td>48527.52339</td>
    </tr>
    <tr>
      <th>42</th>
      <td>EAST OF ENGLAND</td>
      <td>E12000006</td>
      <td>1995-01-01</td>
      <td>56701.59610</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LONDON</td>
      <td>E12000007</td>
      <td>1995-01-01</td>
      <td>74435.76052</td>
    </tr>
    <tr>
      <th>44</th>
      <td>SOUTH EAST</td>
      <td>E12000008</td>
      <td>1995-01-01</td>
      <td>64018.87894</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SOUTH WEST</td>
      <td>E12000009</td>
      <td>1995-01-01</td>
      <td>54705.15790</td>
    </tr>
    <tr>
      <th>47</th>
      <td>England</td>
      <td>E92000001</td>
      <td>1995-01-01</td>
      <td>53202.77128</td>
    </tr>
    <tr>
      <th>48</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-02-01</td>
      <td>82202.77314</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-02-01</td>
      <td>51085.77983</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-02-01</td>
      <td>93190.16963</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-02-01</td>
      <td>64787.92069</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-02-01</td>
      <td>72022.26197</td>
    </tr>
  </tbody>
</table>
</div>



**2.6. Visualizing the data**

To visualize the data, why not subset on a particular London Borough? Maybe do a line plot of Month against Average Price?


```python
df = DF_1 
df['Year'] = df['DATE'].map(lambda t: t.year)
df.head(50)

```

    <ipython-input-59-152e9f9ab58f>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Year'] = df['DATE'].map(lambda t: t.year)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>ID</th>
      <th>DATE</th>
      <th>Average Prices</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98487</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.22660</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.51832</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09036</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.56698</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bromley</td>
      <td>E09000006</td>
      <td>1995-01-01</td>
      <td>81671.47692</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Camden</td>
      <td>E09000007</td>
      <td>1995-01-01</td>
      <td>120932.88810</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croydon</td>
      <td>E09000008</td>
      <td>1995-01-01</td>
      <td>69158.16225</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ealing</td>
      <td>E09000009</td>
      <td>1995-01-01</td>
      <td>79885.89069</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Enfield</td>
      <td>E09000010</td>
      <td>1995-01-01</td>
      <td>72514.69096</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greenwich</td>
      <td>E09000011</td>
      <td>1995-01-01</td>
      <td>62300.10169</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hackney</td>
      <td>E09000012</td>
      <td>1995-01-01</td>
      <td>61296.52637</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hammersmith &amp; Fulham</td>
      <td>E09000013</td>
      <td>1995-01-01</td>
      <td>124902.86020</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Haringey</td>
      <td>E09000014</td>
      <td>1995-01-01</td>
      <td>76287.56947</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harrow</td>
      <td>E09000015</td>
      <td>1995-01-01</td>
      <td>84769.52599</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Havering</td>
      <td>E09000016</td>
      <td>1995-01-01</td>
      <td>68000.13774</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hillingdon</td>
      <td>E09000017</td>
      <td>1995-01-01</td>
      <td>73834.82964</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hounslow</td>
      <td>E09000018</td>
      <td>1995-01-01</td>
      <td>72231.70537</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Islington</td>
      <td>E09000019</td>
      <td>1995-01-01</td>
      <td>92516.48557</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kensington &amp; Chelsea</td>
      <td>E09000020</td>
      <td>1995-01-01</td>
      <td>182694.83260</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kingston upon Thames</td>
      <td>E09000021</td>
      <td>1995-01-01</td>
      <td>80875.84843</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lambeth</td>
      <td>E09000022</td>
      <td>1995-01-01</td>
      <td>67770.98843</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lewisham</td>
      <td>E09000023</td>
      <td>1995-01-01</td>
      <td>60491.26109</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Merton</td>
      <td>E09000024</td>
      <td>1995-01-01</td>
      <td>82070.61330</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Newham</td>
      <td>E09000025</td>
      <td>1995-01-01</td>
      <td>53539.31919</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Redbridge</td>
      <td>E09000026</td>
      <td>1995-01-01</td>
      <td>72189.58437</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Richmond upon Thames</td>
      <td>E09000027</td>
      <td>1995-01-01</td>
      <td>109326.12450</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Southwark</td>
      <td>E09000028</td>
      <td>1995-01-01</td>
      <td>67885.20344</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sutton</td>
      <td>E09000029</td>
      <td>1995-01-01</td>
      <td>71536.97357</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Tower Hamlets</td>
      <td>E09000030</td>
      <td>1995-01-01</td>
      <td>59865.18995</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Waltham Forest</td>
      <td>E09000031</td>
      <td>1995-01-01</td>
      <td>61319.44913</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Wandsworth</td>
      <td>E09000032</td>
      <td>1995-01-01</td>
      <td>88559.04381</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Westminster</td>
      <td>E09000033</td>
      <td>1995-01-01</td>
      <td>133025.27720</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Inner London</td>
      <td>E13000001</td>
      <td>1995-01-01</td>
      <td>78251.97650</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Outer London</td>
      <td>E13000002</td>
      <td>1995-01-01</td>
      <td>72958.79836</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NORTH EAST</td>
      <td>E12000001</td>
      <td>1995-01-01</td>
      <td>42076.35411</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NORTH WEST</td>
      <td>E12000002</td>
      <td>1995-01-01</td>
      <td>43958.48001</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>39</th>
      <td>YORKS &amp; THE HUMBER</td>
      <td>E12000003</td>
      <td>1995-01-01</td>
      <td>44803.42878</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>40</th>
      <td>EAST MIDLANDS</td>
      <td>E12000004</td>
      <td>1995-01-01</td>
      <td>45544.52227</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>41</th>
      <td>WEST MIDLANDS</td>
      <td>E12000005</td>
      <td>1995-01-01</td>
      <td>48527.52339</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>42</th>
      <td>EAST OF ENGLAND</td>
      <td>E12000006</td>
      <td>1995-01-01</td>
      <td>56701.59610</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>43</th>
      <td>LONDON</td>
      <td>E12000007</td>
      <td>1995-01-01</td>
      <td>74435.76052</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>44</th>
      <td>SOUTH EAST</td>
      <td>E12000008</td>
      <td>1995-01-01</td>
      <td>64018.87894</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SOUTH WEST</td>
      <td>E12000009</td>
      <td>1995-01-01</td>
      <td>54705.15790</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>47</th>
      <td>England</td>
      <td>E92000001</td>
      <td>1995-01-01</td>
      <td>53202.77128</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>48</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-02-01</td>
      <td>82202.77314</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-02-01</td>
      <td>51085.77983</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-02-01</td>
      <td>93190.16963</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-02-01</td>
      <td>64787.92069</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-02-01</td>
      <td>72022.26197</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



To limit the number of data points you have, you might want to extract the year from every month value your *Month* column. 

To this end, you *could* apply a ***lambda function***. Your logic could work as follows:
1. look through the `Month` column
2. extract the year from each individual value in that column 
3. store that corresponding year as separate column. 

Whether you go ahead with this is up to you. Just so long as you answer our initial brief: which boroughs of London have seen the greatest house price increase, on average, over the past two decades? 


```python
dfg = df.groupby(by=['London_Borough', 'Year']).mean()
dfg.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Average Prices</th>
    </tr>
    <tr>
      <th>London_Borough</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Havering</th>
      <th>2021</th>
      <td>390734.983483</td>
    </tr>
    <tr>
      <th>Outer London</th>
      <th>1999</th>
      <td>107308.681150</td>
    </tr>
    <tr>
      <th>EAST MIDLANDS</th>
      <th>2014</th>
      <td>151188.957342</td>
    </tr>
    <tr>
      <th>Camden</th>
      <th>2005</th>
      <td>368345.084125</td>
    </tr>
    <tr>
      <th>Ealing</th>
      <th>2012</th>
      <td>318697.611625</td>
    </tr>
    <tr>
      <th>Bromley</th>
      <th>2013</th>
      <td>296669.204058</td>
    </tr>
    <tr>
      <th>Enfield</th>
      <th>2016</th>
      <td>385489.805458</td>
    </tr>
    <tr>
      <th>WEST MIDLANDS</th>
      <th>1995</th>
      <td>49428.563735</td>
    </tr>
    <tr>
      <th>Newham</th>
      <th>2018</th>
      <td>359022.512777</td>
    </tr>
    <tr>
      <th>Brent</th>
      <th>2020</th>
      <td>485851.331367</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfg = dfg.reset_index()
dfg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>London_Borough</th>
      <th>Year</th>
      <th>Average Prices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking &amp; Dagenham</td>
      <td>1995</td>
      <td>51817.969390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>1996</td>
      <td>51718.192690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>1997</td>
      <td>55974.262309</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barking &amp; Dagenham</td>
      <td>1998</td>
      <td>60285.821083</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barking &amp; Dagenham</td>
      <td>1999</td>
      <td>65320.934441</td>
    </tr>
  </tbody>
</table>
</div>



**3. Modeling**

Consider creating a function that will calculate a ratio of house prices, comparing the price of a house in 2018 to the price in 1998.

Consider calling this function create_price_ratio.

You'd want this function to:
1. Take a filter of dfg, specifically where this filter constrains the London_Borough, as an argument. For example, one admissible argument should be: dfg[dfg['London_Borough']=='Camden'].
2. Get the Average Price for that Borough, for the years 1998 and 2018.
4. Calculate the ratio of the Average Price for 1998 divided by the Average Price for 2018.
5. Return that ratio.

Once you've written this function, you ultimately want to use it to iterate through all the unique London_Boroughs and work out the ratio capturing the difference of house prices between 1998 and 2018.

Bear in mind: you don't have to write a function like this if you don't want to. If you can solve the brief otherwise, then great! 

***Hint***: This section should test the skills you acquired in:
- Python Data Science Toolbox - Part One, all modules


```python
def price_ratio(d):
    y1998 = float(d['Average Prices'][d['Year']==1998])
    y2018= float(d['Average Prices'][d['Year']==2018])
    ratio = [y1998/y2018]
    return ratio
```


```python
#Creating a dictionary of the ratios
ratios = {}

for b in dfg['London_Borough'].unique():
    # Subsetting dfg on 'London_Borough'
    borough = dfg[dfg['London_Borough'] == b]
    ratios[b] = price_ratio(borough)

print(ratios) 

```

    {'Barking & Dagenham': [0.20422256235393685], 'Barnet': [0.22945274120785797], 'Bexley': [0.2353507654063011], 'Brent': [0.2043086864360114], 'Bromley': [0.24421308489837312], 'Camden': [0.20261973503252542], 'City of London': [0.18862157770244367], 'Croydon': [0.23803288028014047], 'EAST MIDLANDS': [0.27527471457895053], 'EAST OF ENGLAND': [0.23998652920722383], 'Ealing': [0.23194048191708755], 'Enfield': [0.23455064269011863], 'England': [0.26243599006976326], 'Greenwich': [0.20995010893854218], 'Hackney': [0.16133493530705734], 'Hammersmith & Fulham': [0.24167443054605853], 'Haringey': [0.19475619095546956], 'Harrow': [0.24635417785626296], 'Havering': [0.23120155787014757], 'Hillingdon': [0.23807975835429931], 'Hounslow': [0.25148317824115635], 'Inner London': [0.19339152138506577], 'Islington': [0.20643891170300285], 'Kensington & Chelsea': [0.19675491852791563], 'Kingston upon Thames': [0.23416190234282552], 'LONDON': [0.2136854299558557], 'Lambeth': [0.20170435486140822], 'Lewisham': [0.1835124676472171], 'Merton': [0.21091380604361798], 'NORTH EAST': [0.3535967231925915], 'NORTH WEST': [0.2973168075942841], 'Newham': [0.18848754146121072], 'Outer London': [0.22629811224913093], 'Redbridge': [0.2236545053715767], 'Richmond upon Thames': [0.24967779731157863], 'SOUTH EAST': [0.2612065640720062], 'SOUTH WEST': [0.2634700981993535], 'Southwark': [0.18127484171283462], 'Sutton': [0.24280551426824518], 'Tower Hamlets': [0.2161367227623553], 'WEST MIDLANDS': [0.30199964293728065], 'Waltham Forest': [0.1713867782439487], 'Wandsworth': [0.2101851809159322], 'Westminster': [0.18679140473024677], 'YORKS & THE HUMBER': [0.29796799953704567]}



```python
df_ratios = pd.DataFrame(ratios)
df_ratios.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>City of London</th>
      <th>Croydon</th>
      <th>EAST MIDLANDS</th>
      <th>EAST OF ENGLAND</th>
      <th>...</th>
      <th>SOUTH EAST</th>
      <th>SOUTH WEST</th>
      <th>Southwark</th>
      <th>Sutton</th>
      <th>Tower Hamlets</th>
      <th>WEST MIDLANDS</th>
      <th>Waltham Forest</th>
      <th>Wandsworth</th>
      <th>Westminster</th>
      <th>YORKS &amp; THE HUMBER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.204223</td>
      <td>0.229453</td>
      <td>0.235351</td>
      <td>0.204309</td>
      <td>0.244213</td>
      <td>0.20262</td>
      <td>0.188622</td>
      <td>0.238033</td>
      <td>0.275275</td>
      <td>0.239987</td>
      <td>...</td>
      <td>0.261207</td>
      <td>0.26347</td>
      <td>0.181275</td>
      <td>0.242806</td>
      <td>0.216137</td>
      <td>0.302</td>
      <td>0.171387</td>
      <td>0.210185</td>
      <td>0.186791</td>
      <td>0.297968</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 45 columns</p>
</div>




```python
df_ratios_T = df_ratios.T
df_ratios = df_ratios_T.reset_index()
df_ratios.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking &amp; Dagenham</td>
      <td>0.204223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>0.229453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>0.235351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>0.204309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>0.244213</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Camden</td>
      <td>0.202620</td>
    </tr>
    <tr>
      <th>6</th>
      <td>City of London</td>
      <td>0.188622</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croydon</td>
      <td>0.238033</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EAST MIDLANDS</td>
      <td>0.275275</td>
    </tr>
    <tr>
      <th>9</th>
      <td>EAST OF ENGLAND</td>
      <td>0.239987</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ealing</td>
      <td>0.231940</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Enfield</td>
      <td>0.234551</td>
    </tr>
    <tr>
      <th>12</th>
      <td>England</td>
      <td>0.262436</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Greenwich</td>
      <td>0.209950</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Hackney</td>
      <td>0.161335</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Hammersmith &amp; Fulham</td>
      <td>0.241674</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Haringey</td>
      <td>0.194756</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Harrow</td>
      <td>0.246354</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Havering</td>
      <td>0.231202</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Hillingdon</td>
      <td>0.238080</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ratios.rename(columns={'index':'Borough', 0:'2018'}, inplace=True)
df_ratios.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking &amp; Dagenham</td>
      <td>0.204223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>0.229453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>0.235351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>0.204309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>0.244213</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_ratios= df_ratios.sort_values(by='2018',ascending=False)
top_ratios.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>NORTH EAST</td>
      <td>0.353597</td>
    </tr>
    <tr>
      <th>40</th>
      <td>WEST MIDLANDS</td>
      <td>0.302000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>YORKS &amp; THE HUMBER</td>
      <td>0.297968</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NORTH WEST</td>
      <td>0.297317</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EAST MIDLANDS</td>
      <td>0.275275</td>
    </tr>
    <tr>
      <th>36</th>
      <td>SOUTH WEST</td>
      <td>0.263470</td>
    </tr>
    <tr>
      <th>12</th>
      <td>England</td>
      <td>0.262436</td>
    </tr>
    <tr>
      <th>35</th>
      <td>SOUTH EAST</td>
      <td>0.261207</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Hounslow</td>
      <td>0.251483</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Richmond upon Thames</td>
      <td>0.249678</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Conclusion
What can you conclude? Type out your conclusion below. 

Look back at your notebook. Think about how you might summarize what you have done, and prepare a quick presentation on it to your mentor at your next meeting. 

We hope you enjoyed this practical project. It should have consolidated your data hygiene and pandas skills by looking at a real-world problem involving just the kind of dataset you might encounter as a budding data scientist. Congratulations, and looking forward to seeing you at the next step in the course! 
