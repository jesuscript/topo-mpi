#!/usr/bin/env python

import sys
#import numpy


try:
  import pygtk
  #tell pyGTK, if possible, that we want GTKv2
  pygtk.require("2.0")
except:
  print "Some distributions come with GTK2, but not pyGTK"
  sys.exit(1)

try:
  import gtk
  import gtk.glade
except:
  print "You need to install pyGTK or GTKv2 ",
  print "or set your PYTHONPATH correctly."
  print "try: export PYTHONPATH=",
  print "/usr/local/lib/python2.2/site-packages/"
  sys.exit(1)


class appgui:

  def __init__(self):
    """
    In this init we are going to display the main
    serverinfo window
    """
    
    import sys
    sys.path.append('/home/antolikjan/topographica/topographica/contrib')
    
    import dd
    self.db = dd.DB()
    self.db.load_db("../topographica/modelfitDB.dat")
    
    gladefile="data_browser.glade"
    windowname="window1"
    self.wTree=gtk.glade.XML(gladefile,windowname)
    dic = { "on_button1_clicked" : 
            self.button1_clicked,
            "on_serverinfo_destroy" : 
            (gtk.mainquit) ,
	    
	    "on_treeview1_row_activated" :
	    self.raw_selected
	    
	    
	    }
    self.wTree.signal_autoconnect(dic)
    
    
    import gobject
    self.treeview=self.wTree.get_widget("treeview1")
    self.treemodel=gtk.TreeStore(gobject.TYPE_INT,gobject.TYPE_STRING,
                                 gobject.TYPE_STRING)
    self.treeview.set_model(self.treemodel)

    self.treeview.set_headers_visible(True)
    renderer=gtk.CellRendererText()
    column=gtk.TreeViewColumn("ID",renderer, text=0)
    column.set_resizable(True)
    self.treeview.append_column(column)
    renderer=gtk.CellRendererText()
    column=gtk.TreeViewColumn("Parameters",renderer, text=1)
    column.set_resizable(True)
    self.treeview.append_column(column)
    renderer=gtk.CellRendererText()
    column=gtk.TreeViewColumn("Description",renderer,text=2)
    column.set_resizable(gtk.TRUE)
    self.treeview.append_column(column)
    self.treeview.show()
    self.insert_db()
	
    
    self.funcs = {
    	"ReversCorrelationRFs" : self.showRFs 
	#"ReversCorrelationCorrectPercentage"
	#"ReversCorrelationTFCorrectPercentage"
	#"ReversCorrelationPredictedActivities"
	#"ReversCorrelationPredictedActivities+TF"
	#"ReversCorrelationPredictedValidationActivities"
	#"ReversCorrelationPredictedValidationActivities+TF"
	#"ReversCorrelationNormalizedErrors"
	#"ReversCorrelationNormalizedErrors+TF"
	#"ReversCorrelationCorrCoefs"
	#"ReversCorrelationCorrCoefs+TF"
	#"ReversCorrelationTransferFunction"
    }
    

    
    return

#####CALLBACKS
  def button1_clicked(self,widget):
    print "button clicked"

  def raw_selected(self,model,path,trash):
	  idd = self.treemodel.get_value(self.treemodel.get_iter(path),0)
	  
	  if idd != -1:
	     data_name = self.db.db[idd][1]	  
	     funcs[data_name](self.db.db[idd][2])


  def insert_db(self):
      mem = {}
      for (a,b,c) in self.db.db:
                if not mem.has_key(str(a)):
		   res = self.db.find(a,None)	
		   print len(res)		
		   myiter=self.treemodel.insert_after(None,None)
		   self.treemodel.set_value(myiter,1,a)
  		   self.treemodel.set_value(myiter,0,-1)
		   for (t,r,t1,idd) in res:
		       z = self.treemodel.insert_after(myiter,None)
		       self.treemodel.set_value(z,2,r)	   
		       self.treemodel.set_value(z,0,idd)
    		   mem[str(a)]=True
		
  def showRFs(self,RFs):
       (x,y) = numpy.shape(RFs)	   
       m = numpy.max(numpy.abs(numpy.min(RFs)),numpy.abs(numpy.max(RFs)))	    
       pylab.figure()
       pylab.title("Receptive fields", fontsize=16)
       for i in xrange(0,x):
           pylab.subplot(15,15,i+1)
           w = numpy.array(Z[i]).reshape(numpy.sqrt(y),numpy.sqrt(y))
           pylab.show._needmain=False
           pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
       pylab.show()

	   
# we start the app like this...
app=appgui()
gtk.main()
