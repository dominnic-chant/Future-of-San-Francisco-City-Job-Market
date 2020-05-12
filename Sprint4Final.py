"""
San Francisco Employee Compensation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import seaborn as sns
import statsmodels.api as sn
from sklearn.linear_model import LinearRegression

#This creates the directory all the graphs will be saved under
#os.mkdir('SFEmpComp')

#####  EDA  #####
#%% '''ScatterPlot Graphing'''
def ScatterPlotMatrix():
    #This plots the scatterplot in a matrix form
    #The plot is colored from blue at the lowest Total Salary to yellow at the highest Total Salary
    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()
    print("Creating a table with filtered data for the scatterplot graph using SQL")
    SQLCurr.execute("""SELECT org_code, org, total_salary, total_benefit
                        FROM SF_EMP_COMP.comp
                        WHERE total_salary>=0 and total_benefit>=0 and year_type = 'Fiscal' and year <=2019 and org_code<=6;""")
    SCPlot = pd.DataFrame(data = SQLCurr.fetchall(), columns = ['org_code', 'org', 'total_salary', 'total_benefit']).set_index('org')   
    
        #This creates a linear line of Total Benefits= 0.3*Total Compensation to create a visualization for the hypothesis testing
        #658867 is chosen as the limit as this is the highest value in the dataset
    xBnftSlry = np.linspace(0,658867)
    yBnftSlry = 0.3*(np.linspace(0,808867))
    
        #This function creates a variable for each Organization Group and the corresponding Total Salary and Total Benefits
    OrgGroup = ['Public Protection', 'Public Works Transportation Commerce', 'Human Welfare and Neighborhood Development', 'Community Health', 'Culture and Recreation', 'General Administration and Finance']
    OrgGroupPos = 0
    while OrgGroupPos<=5:
        OrgGrouper = SCPlot[SCPlot['org_code'] == OrgGroupPos+1]
        OrgGrouperXval = OrgGrouper['total_salary']
        OrgGrouperYval = OrgGrouper['total_benefit']
        plt.scatter(OrgGrouperXval, OrgGrouperYval, 5, alpha=0.5, c = OrgGrouper['total_salary'], label='CompRecieved')
        plt.plot(xBnftSlry, yBnftSlry, 'r--', color='r', label='30% Total Compensation')
        plt.title(OrgGroup[OrgGroupPos])
        plt.xlabel('Total Salary')
        plt.ylabel('Total Benefits')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlim(0, 658867)
        plt.ylim(0, 150000)
        plt.savefig(OrgGroup[OrgGroupPos] + '.png', bbox_inches='tight')
        # plt.show()
        OrgGroupPos += 1

#%% '''Job Count Prediction'''  
def JobCountPrediction():
    #This creates a dataframe using SQL
    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()
    print("Creating a table with filtered data for the barplot graph using SQL")
    SQLCurr.execute("""SELECT org_code as OrganizationGroup, (ct2019-ct2013)/7 as gradient, ct2013 as linear, ct2019
                        FROM
                            (SELECT org_code, count(job) as ct2013
                            FROM SF_EMP_COMP.comp
                            WHERE total_salary>=0 and total_benefit>=0 and year_type = 'Fiscal' and year=2013
                            GROUP BY 1) as lhs
                        INNER JOIN
                            (SELECT org_code, count(job) as ct2019
                            FROM SF_EMP_COMP.comp
                            WHERE total_salary>=0 and total_benefit>=0 and year_type = 'Fiscal' and year=2019
                            GROUP BY 1) as rhs
                        USING (org_code)
                        WHERE org_code<=6;""")                        
    df = pd.DataFrame(data = SQLCurr.fetchall(), columns = ['OrganizationGroup', 'Gradient', 'YIntercept', 'Ct2019']).set_index('OrganizationGroup')
    #This creates a barplot from 2013 to 2019 (first and last year with complete data) of each Organization Group with its corresponding Total Count of Employees
    fig, ax = plt.subplots()  
    OrgGroup = ['Protection', 'PubWorks', 'HumanWelfare', 'Health', 'Culture', 'Gen Admin']
    OrgGroupPos = 0
    yearprediction = 2025

    #predictiongr - the predicted future value 
    #increasect/decreasect - the values changed over the years
    while OrgGroupPos<=5:
        yender = (df.iloc[OrgGroupPos, 0]*(yearprediction-2013))+df.iloc[OrgGroupPos, 1]
        if (df.iloc[OrgGroupPos, 2]>df.iloc[OrgGroupPos, 1]):
            predictiongr = plt.bar(x = OrgGroup[OrgGroupPos], height = yender-df.iloc[OrgGroupPos, 2], width=0.5, bottom=df.iloc[OrgGroupPos, 2] , align='center', color='Orange', alpha=0.5)
            increasect = plt.bar(x = OrgGroup[OrgGroupPos], height = df.iloc[OrgGroupPos, 2]-df.iloc[OrgGroupPos, 1], width=0.5, bottom=df.iloc[OrgGroupPos, 1], align='center', color='Lime')
        elif (df.iloc[OrgGroupPos, 2]<=df.iloc[OrgGroupPos, 1]):
            decreasect = plt.bar(x = OrgGroup[OrgGroupPos], height = df.iloc[OrgGroupPos, 2]-df.iloc[OrgGroupPos, 1], width=0.5, bottom=df.iloc[OrgGroupPos, 2], align='center', color='Red')
            predictiongr = plt.bar(x = OrgGroup[OrgGroupPos], height = yender-df.iloc[OrgGroupPos, 2], width=0.5, bottom=yender, align='center', color='Orange', alpha=0.5)
        OrgGroupPos += 1
    
    #These sets the legend and titles for the barplot graph
    ax.set_ylabel('Total Count of Employees')
    ax.set_xlabel('Organization Group')
    plt.legend((increasect, predictiongr), (('Increased'), ('Predicted No.Employees in '+str(yearprediction))))
    fig.tight_layout()
    plt.tick_params(axis='x',labelsize=8.5)
    fig1 = plt.gcf()
    # plt.show()
    fig1.savefig('PredictionBarPlot.png')

#%% '''Line Graphs''' 
def plotLines(df, year):
    plt.figure()
    plt.plot(df.loc[(df.org == 'Community Health') & (df.year == year), ['perc_emp']], df.loc[(df.org == 'Community Health') & (df.year == year), ['perc_comp']], color = 'blue', label = 'Health')
    plt.plot(df.loc[(df.org == 'Culture & Recreation') & (df.year == year), ['perc_emp']], df.loc[(df.org == 'Culture & Recreation') & (df.year == year), ['perc_comp']], color = 'green', label = 'Culture')
    plt.plot(df.loc[(df.org == 'General Administration & Finance') & (df.year == year), ['perc_emp']], df.loc[(df.org == 'General Administration & Finance') & (df.year == year), ['perc_comp']], color = 'red', label = 'General Admin')
    plt.plot(df.loc[(df.org == 'Human Welfare & Neighborhood Development') & (df.year == year), ['perc_emp']], df.loc[(df.org == 'Human Welfare & Neighborhood Development') & (df.year == year), ['perc_comp']], color = 'orange', label = 'Human Welfare')
    plt.plot(df.loc[(df.org == 'Public Protection') & (df.year == year), ['perc_emp']], df.loc[(df.org == 'Public Protection') & (df.year == year), ['perc_comp']], color = 'black', label = 'Pub Protection')
    plt.plot(df.loc[(df.org == 'Public Works, Transportation & Commerce') & (df.year == year), ['perc_emp']], df.loc[(df.org == 'Public Works, Transportation & Commerce') & (df.year == year), ['perc_comp']], color = 'cyan', label = 'Pub Works')

    plt.title(year)
    plt.legend(loc = 'upper left', fontsize = '10') 

def generate_graphs(df):
    for i in range(2013,2020): 
        plotLines(df, i)
        plt.savefig('LineGraph' + str(i) + '.png', dpi = 200)
    # plt.show()

def queryLineGraph():
    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"

    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()

    SQLCurr.execute("""with temp as 
                    (select org, total_emp, total_comp, count(empid) as ct, sum(_comp) as sm, ntile as percentiles, year
                    from
                        (select org, empid, total_emp, total_comp, _comp, year,
                                ntile(4) over(partition by org, year
                                              order by _comp asc)
                        from
                           (select org, empid, year_type, year, total_comp as _comp,
                                    sum(total_comp) over(partition by org, year) as total_comp,
                                    count(empid) over(partition by org, year) as total_emp
                            from SF_EMP_COMP.comp
                            where (year_type = 'Fiscal') and (year < 2020) and (org != 'General City Responsibilities')) as oq
                        order by org, year asc) as oq2
                    group by 1,2,3,6,7
                    order by org, ntile)
            
              select org, year,
                     round((((sum(ct) over(partition by org, year 
                                           order by percentiles)) / total_emp) * 100)::numeric, 2) as perc_emps,
                     round((((sum(sm) over(partition by org, year 
                                           order by percentiles)) / total_comp) * 100)::numeric, 2) as perc_comp
              from temp
              order by year, org, perc_emps
              ;""")

    df = pd.DataFrame(data = SQLCurr.fetchall(),
            columns = ['org','year', 'perc_emp', 'perc_comp'])

    return df

def LineGraphs():
    df = queryLineGraph()
    generate_graphs(df)

############

#%% '''Distinct Job HeatMap'''    
def DistinctJobHeatMap():
    #This creates a dataframe using SQL
    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()
    print("Creating a Heatmap Table in SQL")
    SQLCurr.execute("""SELECT OrganizationGroup, year, round((jobct::float/avgjobct)*100)::int as prctchnge
                        FROM
                            (SELECT org_code as OrganizationGroup, year, count(job) as jobct
                            FROM SF_EMP_COMP.comp
                            WHERE total_salary>=0 and total_benefit>=0 and year_type = 'Fiscal' and year <=2019
                            GROUP BY 1,2) as lhs
                        INNER JOIN
                            (SELECT OrganizationGroup, avg(jobct) as avgjobct
                            FROM
                                (SELECT org_code as OrganizationGroup, year, count(job) as jobct
                                FROM SF_EMP_COMP.comp
                                WHERE total_salary>=0 and total_benefit>=0 and year_type = 'Fiscal' and year <=2019
                                GROUP BY 1,2) as inners
                            GROUP BY 1) as rhs
                        using(OrganizationGroup)
                        WHERE OrganizationGroup<=6;""")                        
    JobHeatMap = pd.DataFrame(data = SQLCurr.fetchall(), columns = ['OrganizationGroup', 'year', 'prctchnge'])
    
    #This creates a barplot from 2013 to 2019 (first and last year with complete data) of each Organization Group with its corresponding Total Count of Employees
    fig, ax = plt.subplots()    
    JobHeatMap = JobHeatMap.pivot('OrganizationGroup', 'year', 'prctchnge')
    ax = sns.heatmap(JobHeatMap, linewidths=.5, center=100, cmap="Greens", annot = True, fmt="d")
    ax.collections[0].colorbar.set_label("Percentage Change")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    fig.savefig("DistinctJobHeatBluesMap.png")

#%% '''Hypothesis Testing'''
def HypothesisTesting():
    #This plots the scatterplot in a matrix form
    #The plot is colored from blue at the lowest Total Salary to yellow at the highest Total Salary
    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()
    print("Creating a table with filtered data for hypothesis testing on the scatterplot graph using SQL")
    SQLCurr.execute("""SELECT org_code, org, total_salary, total_benefit
                        FROM SF_EMP_COMP.comp
                        WHERE total_salary>10000 and total_benefit>0 and year_type = 'Fiscal' and year<=2019
                        and org_code<=6;;""")
    SCPlot = pd.DataFrame(data = SQLCurr.fetchall(), columns = ['org_code', 'org', 'total_salary', 'total_benefit']).set_index('org')   
    OrgGroup = ['Public Protection', 'Public Works Transportation Commerce', 'Human Welfare and Neighborhood Development', 'Community Health', 'Culture and Recreation', 'General Administration and Finance']
    
    OrgGroupPos = 0
    while OrgGroupPos<=5:
        OrgGrouper = SCPlot[SCPlot['org_code'] == OrgGroupPos+1]
        salarymean = OrgGrouper['total_salary'].mean()
        hypotest = sn.stats.ztest(OrgGrouper['total_benefit'], value = 3*salarymean/7, alternative = 'smaller')
        pval = hypotest[1]
        if (pval>0.95):
            print(OrgGroup[OrgGroupPos], 'has at least a 95% confidence interval that the total_benefits is greater than 30% total compensation.')
            print(OrgGroup[OrgGroupPos], 'has', pval, 'confidence.')
            print('')
        OrgGroupPos += 1

#%% '''Query HeatMap'''
# this returns a heatmap of 2018
# shows jobs that must exist in all orgs
# these are the top 12 compensated jobs
# compares each job to its counterparts in other orgs
# the darker it is, the less compensated it is compared to its counterparts
# the lighter it is, the more compensated

def queryHeatMap():
    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"

    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()

    SQLCurr.execute("""with iq as 
    					(select health.job, 
    		                    health.avgcomp as health,
    		                    pubprotect.avgcomp as pubprotect, 
    		                    pubworks.avgcomp as pubworks, 
    		                    genadmin.avgcomp as genadmin, 
    		                    culture.avgcomp as culture, 
    		                    welfare.avgcomp as welfare
    		            from
    		                (select org, job, avg(total_comp) as avgcomp
    		                from
    		                    (select org, job, total_comp, year
    		                    from SF_EMP_COMP.comp
    		                    where (year_type = 'Fiscal') and (year = 2018)
    		                    order by org, job, year, total_comp) as iq
    		                where (org = 'Community Health')
    		                group by org, job) as health
    		            join
    		                (select org, job, avg(total_comp) as avgcomp
    		                from
    		                    (select org, job, total_comp, year
    		                    from SF_EMP_COMP.comp
    		                    where (year_type = 'Fiscal') and (year = 2018)
    		                    order by org, job, year, total_comp) as iq
    		                where (org = 'Public Protection')
    		                group by org, job) as pubprotect
    		            
    		            on(health.job = pubprotect.job)
    		            join
    		                (select org, job, avg(total_comp) as avgcomp
    		                from
    		                    (select org, job, total_comp, year
    		                    from SF_EMP_COMP.comp
    		                    where (year_type = 'Fiscal') and (year = 2018)
    		                    order by org, job, year, total_comp) as iq
    		                where (org = 'Public Works, Transportation & Commerce')
    		                group by org, job) as pubworks
    		            
    		            on(health.job = pubworks.job)
    		            join
    		                (select org, job, avg(total_comp) as avgcomp
    		                from
    		                    (select org, job, total_comp, year
    		                    from SF_EMP_COMP.comp
    		                    where (year_type = 'Fiscal') and (year = 2018)
    		                    order by org, job, year, total_comp) as iq
    		                where (org = 'General Administration & Finance')
    		                group by org, job
    		                order by job) as genadmin
    		            
    		            on(health.job = genadmin.job)
    		            join
    		                (select org, job, avg(total_comp) as avgcomp
    		                from
    		                    (select org, job, total_comp, year
    		                    from SF_EMP_COMP.comp
    		                    where (year_type = 'Fiscal') and (year = 2018)
    		                    order by org, job, year, total_comp) as iq
    		                where (org = 'Culture & Recreation')
    		                group by org, job
    		                order by job) as culture
    		            
    		            on(health.job = culture.job)
    		            join
    		                (select org, job, avg(total_comp) as avgcomp
    		                from
    		                    (select org, job, total_comp, year
    		                    from SF_EMP_COMP.comp
    		                    where (year_type = 'Fiscal') and (year = 2018)
    		                    order by org, job, year, total_comp) as iq
    		                where (org = 'Human Welfare & Neighborhood Development')
    		                group by org, job
    		                order by job) as welfare
    		            
    		            on(health.job = welfare.job)
    		            order by 1,2)

    				    select job, 
    				        round(100 * (health /  ((health + pubprotect + pubworks + genadmin + culture + welfare) / 7))::numeric, 0) as hperc,
    				        round(100 * (pubprotect /  ((health + pubprotect + pubworks + genadmin + culture + welfare) / 7))::numeric, 0) as ppperc,
    				        round(100 * (pubworks /  ((health + pubprotect + pubworks + genadmin + culture + welfare) / 7))::numeric, 0) as pwperc,
    				        round(100 * (genadmin /  ((health + pubprotect + pubworks + genadmin + culture + welfare) / 7))::numeric, 0) as gaperc,
    				        round(100 * (culture /  ((health + pubprotect + pubworks + genadmin + culture + welfare) / 7))::numeric, 0) as cperc,
    				        round(100 * (welfare /  ((health + pubprotect + pubworks + genadmin + culture + welfare) / 7))::numeric, 0) as wperc 
    				    from iq;""")
    df = pd.DataFrame(data = SQLCurr.fetchall(), columns = ['job', 'H', 'PP', 'PW', 'GA', 'C', 'W'])
    df.loc[ : , 'temp'] = (df.H + df.PP + df.PW + df.GA + df.C + df.W)
    df = (df.sort_values(['temp'], ascending = False)
            .loc[(df.temp > 700), ['job', 'PW', 'PP', 'H', 'GA', 'C', 'W']]
            .sort_values(['job'])
    		.set_index('job')
    		.apply(pd.to_numeric))
    
    return(df)

def GenerateHeatMap(df):
    plt.figure(figsize=(7, 7))
    plt.title('2018')
    ax = sns.heatmap(df, annot = True, annot_kws= {'size' : 9}, square = True, cbar_kws = {'label' : 'Percent Change'}, fmt = 'g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set(xlabel = 'Organizations', ylabel = 'Jobs')
    # ax.figure.tight_layout()
    ax.figure.subplots_adjust(left = 0.4)
    plt.savefig('JobsHeatMap.png', dpi = 200)
    
#%% '''Generate HeatMap Regression'''
#this returns 10 rows and 5 columns 
#predicts the year 2025, 2030, 2040
#returns the top 10 paying jobs in the future


def queryRegression(orgName):
    # print("querying for " + orgName + "...")

    CONNECTION_STRING = "dbname='bsdsclass' user='' host='bsds200.c3ogcwmqzllz.us-east-1.rds.amazonaws.com' password=''"

    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCurr = SQLConn.cursor()

    SQLCurr.execute("""
                select total_comp, org, job, year
                from SF_EMP_COMP.comp
                where (org = '%s') and (year_type = 'Fiscal') and (total_salary > 0) and (total_benefit > 0)
                order by job
                ;""" % (orgName))

    df = pd.DataFrame(data = SQLCurr.fetchall(), columns = ['total_comp','org', 'job', 'year'])
    return df

def regression(df, yr):
	
    answer = df.copy()
    # print("creating dummies...")
    dummies = pd.get_dummies(df.job)
    df = df.join(dummies)
    df = df.drop(['job', 'org'], axis = 1)
    reg = LinearRegression()
    
    x = df.iloc[: , 1:]
    y = df['total_comp']
    
    reg.fit(x, y)
    
    numIndVar = x.shape[1]

    answer = (answer.loc[ : , ['org', 'job']].drop_duplicates())
    year = yr
    jobCompPredicted = []
    temp = [0] * numIndVar
    temp[0] = year
 #this for loop creates an array of values that corresponds to the betas in regression equation
 #only the first value is fixed at the year we're predicting, everything else is either 1 or 0, with 1 being the job we're predicting for 
    for x in range(1, numIndVar):
    	temp[x] = 1
    	temp[x - 1] = 0
    	temp[0] = year
    	predicted = round(float(reg.predict([temp])[0]), 2)
    	jobCompPredicted.append(predicted)
    answer = answer.dropna()
    answer.loc[ : , 'year'+str(year)] = jobCompPredicted
    return answer

def JobCompRegression():
    FutureYears = [2025, 2030, 2040]
    for year in FutureYears:
    #doing this one outside the loop to have an allOrgsDf for the other orgs to append to 
        orgName = 'Community Health'
        regQuery = queryRegression(orgName)
        allOrgsDf = regression(regQuery, year)

        OrgGroup = ['Public Protection', 'Public Works, Transportation & Commerce', 'General Administration & Finance', 'Culture & Recreation', 'Human Welfare & Neighborhood Development']
        for i in OrgGroup:
            regQuery = queryRegression(i)
            singleOrgdf = regression(regQuery, year)
            allOrgsDf = pd.concat([allOrgsDf, singleOrgdf], axis = 0)
        allOrgsDf = (allOrgsDf.sort_values("year" + str(year), ascending = False).head(10).reset_index(drop = True))
        print(allOrgsDf)

def sprint4():
    # ScatterPlotMatrix()
    # JobCountPrediction()
    LineGraphs()
    # DistinctJobHeatMap()
    # HypothesisTesting()
    # GenerateHeatMap(queryHeatMap())
    # JobCompRegression()

sprint4()
