from flask import Flask, render_template
from graph import build_graph
import matplotlib.pyplot as plt
 
app = Flask(__name__)

@app.route('/graphs')
def graphs():
    #These coordinates could be stored in DB
    x1 = [0, 1, 2, 3, 4]
    y1 = [10, 30, 40, 5, 50]
    x2 = [0, 1, 2, 3, 4]
    y2 = [50, 30, 20, 10, 50]
    x3 = [0, 1, 2, 3, 4]
    y3 = [0, 30, 10, 5, 30]

    plt.plot([0, 0.1, 0.2, 0.3])
    plt.ylabel('Stenosis Trend')
    axis([0,750,25,100])
    plt.show()
 
    graph0_url = build_graph([x1,y1],[x2,y2]);

    graph1_url = build_graph(x1,y1);
    graph2_url = build_graph(x2,y2);
    graph3_url = build_graph(x3,y3);
 
    return render_template('graphs.html', graph12=graph12_url, graph1=graph1_url, graph2=graph2_url, graph3=graph3_url)
 
if __name__ == '__main__':
    app.debug = True
    app.run()
 