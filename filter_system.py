"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter
        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.process_noise = Q # 4x4 prediction noise matrix
        self.measurement_noise = R # 2x2 measurement noise matrix
        self.Dt = np.zeros((4,4)) # Dt state transition matrix initialized with zeros
        self.dt = 1 # delta t for dynamics equations
        self.Dt = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
        self.Mt = np.zeros((2,4)) # Mt measurement matrix initialized with zeros
        self.Mt = np.array([[1,0],[0,1],[0,0],[0,0]])
        self.Mt = np.transpose(self.Mt)
        self.covariance = np.identity(4) # not sure what the diagonal value should be
        
#        raise NotImplementedError

    def predict(self):
    
        #print("self state: ",self.state)
        self.state = np.dot(self.Dt,self.state)
        #print("state update: ",self.state)
        self.covariance = np.add(np.dot(self.Dt,np.dot(self.covariance,np.transpose(self.Dt))),self.process_noise)
#        raise NotImplementedError

    def correct(self, meas_x, meas_y):
    
        K = np.add(np.dot(self.Mt,np.dot(self.covariance,np.transpose(self.Mt))),self.measurement_noise)
        K = np.linalg.inv(K)
        K = np.dot(np.transpose(self.Mt),K)
        K = np.dot(self.covariance,K)
        
        vx = meas_x - self.state[0]
        vy = meas_y - self.state[1]
        X = np.array([meas_x,meas_y,vx,vy])
        #print("meas x: ",meas_x)

        temp_mat = np.dot(self.Mt,self.state)
        Y = np.array([X[0],X[1]])
        temp_mat = np.subtract(Y,temp_mat)
        temp_mat = np.dot(K,temp_mat)
        self.state = np.add(self.state,temp_mat)
        #self.state = np.add(self.state,np.dot(K,np.subtract(X,np.dot(self.Mt,self.state))))
        
        self.covariance = np.dot(np.subtract(np.identity(4),np.dot(K,self.Mt)),self.covariance)
        
#        raise NotImplementedError

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.
    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.
        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.
        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        #self.sigma_exp = 0.1*self.sigma_exp
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        self.particles = (np.random.rand(self.num_particles,2) * np.shape(self.frame)[0]).astype(int) # Initialize your particles array. Read the docstring.
        #print("init particles: ",self.particles)
        self.weights = np.random.uniform(size=self.num_particles)  # Initialize your weights array. Read the docstring.
        #self.weights = [1/self.num_particles] * self.num_particles
        #print("initial weights: ",self.weights)
        # Initialize any other components you may need when designing your filter.

        #raise NotImplementedError

    def get_particles(self):
        """Returns the current particles state.
        This method is used by the autograder. Do not modify this function.
        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.
        This method is used by the autograder. Do not modify this function.
        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.
        Returns:
            float: similarity value.
        """

        #print("template: ",template)
        #print("frame cutout: ",frame_cutout)
        #print("err check: ",((template-frame_cutout)**2))

        img_template = np.copy(template)
        img_template = cv2.cvtColor(img_template,cv2.COLOR_BGR2GRAY)
        
        img_frame_cutout = np.copy(frame_cutout)
        img_frame_cutout = cv2.cvtColor(img_frame_cutout,cv2.COLOR_BGR2GRAY)




        error = ((img_template-img_frame_cutout)**2).mean(axis=None)
        #error = ((img_template-img_frame_cutout)**2).sum(axis=None)
        #error = sum(sum(sum((template-frame_cutout)**2)))
        
        
        #print("error: ",error)
        #error = ((template-frame_cutout)**2).sum()

        return error
        
        #return NotImplementedError

    def resample_particles(self):
        """Returns a new set of particles
        This method does not alter self.particles.
        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.
        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """

        fin = np.random.multinomial(self.num_particles,self.weights)

        particles = self.particles

        new_particles = []
        for i in range(len(fin)):
            if fin[i] >= 1:
                #new_particles.append(particles[i])
                for n in range(fin[i]):
                    new_particles.append(particles[i])
        #print("fin: ",fin)

        new_particles = np.stack(new_particles)
        #print("length: ",len(new_particles))

        for n in range(len(new_particles)):
            if n % 5 == 0:
                new_particles[n][0] += np.random.randint(-self.sigma_exp,self.sigma_exp)
                new_particles[n][1] += np.random.randint(-self.sigma_exp,self.sigma_exp)


        #noise1 = np.random.normal(-10,10,len(new_particles))
        #[int(x) for x in noise1]
        #noise2 = np.random.normal(-10,10,len(new_particles))
        #[int(x) for x in noise2]

        #print("new particles dtype: ",new_particles.dtype)
        #print("noise dtype: ",noise.dtype)


        #new_particles += noise

        return new_particles
        #return fin

        #return NotImplementedError

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.
        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.
        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        img1 = np.copy(frame)
        img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        #print("process frame shape: ",np.shape(img1))

        temp = self.template

        img_temp = np.copy(temp)
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2GRAY)

        temp_size_y = np.shape(temp)[0]
        temp_size_x = np.shape(temp)[1]

        temp_size_y = np.shape(img_temp)[0]
        temp_size_x = np.shape(img_temp)[1]

        particles = self.particles
        #print("sig dyn: ",self.sigma_dyn)

        weights = self.weights
        #print("particles: ",particles)
        #print("old weights: ",weights)
        #self.sigma_exp = 5

        for i in range(len(particles)):
            #print("i: ",i)
            #frame_cutout = img1[int(particles[i][1]-int(temp_size_y/2)):int(particles[i][1]+int(temp_size_y/2)),int(particles[i][0]-int(temp_size_x/2)):int(particles[i][0]+int(temp_size_x/2)),:]
            frame_cutout = img1[int(particles[i][1]-int(temp_size_y/2)):int(particles[i][1]+int(temp_size_y/2)+1),int(particles[i][0]-int(temp_size_x/2)):int(particles[i][0]+int(temp_size_x/2)+1),:]
            #print("frame cutout: ",frame_cutout)
            if np.shape(temp) == np.shape(frame_cutout):
                #print("here1")
                mse = self.get_error_metric(temp,frame_cutout)
                weights[i] = np.exp(-mse/(2*self.sigma_exp))
            else:
                weights[i] = 0
                #print("here2")
                #print("cutout shape: ",np.shape(frame_cutout))
                #temp2 = temp[0:np.shape(frame_cutout)[0],0:np.shape(frame_cutout)[1],:]
                #print("temp2: ",temp2)
                #print("len check: ",np.shape(temp2))
                #for n in range(len(temp2[0])):
                #    print("temp2n: ",temp2[0][n])
                #    temp2[0][n]= [temp2[0][n][0],temp2[0][n][2],temp2[0][n][1]]
                ##print("template2: ",temp2)
                ##print("frame cutout: ",frame_cutout)
                #mse = self.get_error_metric(temp2,frame_cutout)

            #print("mse: ",mse)

            tot_weights = sum(weights)

            for n in range(len(weights)):
                weights[n] = weights[n] / tot_weights
            #print("weights sum: ",sum(weights))
            
            #weights[i] = np.exp(-mse/(2*self.sigma_exp))




        #print("old weights: ",self.weights)
        #print("new weights: ",weights)

        self.weights = weights

        #print("old particles: ",self.particles)
        #print("old particles shape: ",np.shape(self.particles))


        new_particles = self.resample_particles()
        #new_particles = np.stack(new_particles)


        #print("new particles: ",new_particles)
        #print("new particles shape: ",np.shape(new_particles))
        
        self.particles = new_particles



    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        #print("particles for render: ",self.particles)

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        rect_min_x = int(x_weighted_mean - ((np.shape(self.template)[1])/2))
        #print('rect min x: ',rect_min_x)
        rect_min_y = int(y_weighted_mean - ((np.shape(self.template)[0])/2))
        rect_max_x = int(x_weighted_mean + ((np.shape(self.template)[1])/2))
        rect_max_y = int(y_weighted_mean + ((np.shape(self.template)[0])/2))

        cv2.rectangle(frame_in,(rect_min_x,rect_min_y),(rect_max_x,rect_max_y),(0,0,255),2)

        particles = self.particles
        #print("particles render: ",particles)

        euc_dist_weighted = []
        for i in range(len(particles)):
            cv2.circle(frame_in,(particles[i][0],particles[i][1]),1,(0,0,255),5)
            euc_dist_weighted.append(((((particles[i][0] - x_weighted_mean) ** 2) + ((particles[i][0] - x_weighted_mean) ** 2)) ** 0.5)*self.weights[i])
            #frame_in[i[1],i[0]] = 

        euc_dist_weighted_sum = int(sum(euc_dist_weighted))
        
        cv2.circle(frame_in,(int(x_weighted_mean),int(y_weighted_mean)),euc_dist_weighted_sum,(0,0,255),3)

        #raise NotImplementedError



class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.
        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.
        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].
        Returns:
            None.
        """

        #print("particles appearance: ",self.particles)

        template = self.template
        #frame = self.frame

        temp_y = np.shape(template)[0]
        temp_x = np.shape(template)[1]

        particles = self.particles
        #particles_mean = particles.mean(axis=0)
        particles_mean = np.average(particles,axis=0,weights=self.weights)
        #best_weight = max(self.weights)
        #print("t1: ",best_weight)
        #best_weight = np.where(self.weights==best_weight)
        #print("t2: ",best_weight)
        #best_weight = list(best_weight)[0][0]
        #print("t3: ",best_weight)
        #particles_mean = particles[best_weight]
        

        weights = self.weights
        #print("weights: ",weights)

        #print("template: ",np.shape(template))
        cv2.imwrite('template.png',template)

        #print("particles: ",particles)
        #print("test: ",particles_mean[1])
        best = frame[int(particles_mean[1]-(temp_y/2)):int(particles_mean[1]+(temp_y/2)),int(particles_mean[0]-(temp_x/2)):int(particles_mean[0]+(temp_x/2)),:]

        best = cv2.cvtColor(best,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('best.png',best)
        template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        
        if np.shape(best) == np.shape(template):
            new_template = self.alpha*best + (1-self.alpha)*template
        else:
            new_template = template

        new_template = new_template.astype('float32')
        new_template = cv2.cvtColor(new_template,cv2.COLOR_GRAY2BGR)

        #print("new template: ",np.shape(new_template))
        cv2.imwrite('new_template.png',new_template)

        self.template = new_template.astype('float32')

        #print("particle mean: ",self.particles.mean(axis=0))
        #print("particle mean2: ",self.particles.mean(axis=1))

        #for x in frame:
        #    for y in x:

        #        frame_cutout = frame[int(y-(temp_y/2)):int(y+(temp_y/2)),int(x-(temp_x/2)):int(x+(temp_x/2)),:]


        #ParticleFilter.get_error_metric(self, template, frame_cutout)
        #prev_particles = self.particles
        #print("prev particles: ",prev_particles)

        #self.sigma_exp = 0.8*self.sigma_exp
        #self.sigma_exp = 20
        #self.num_particles = 5*self.num_particles
        ParticleFilter.process(self,frame)
        #print("here")

        #new_best = self.alpha*self.particles + (1-self.alpha)*prev_particles

        #self.particles = new_best
        #conv_part = self.particles
        #conv_part = conv_part.astype(int)
        #self.particles = conv_part

        #print("conv part: ",conv_part)
        #for i in range(len(conv_part)):
        #    print("conv part: ",type(conv_part[i]))
        #    print("self particles1: ",conv_part[i][0])
        #    conv_part[i] = (conv_part[i]).astype(int)
        #    #self.particles[i][0] = int(self.particles[i][0])
        #    #self.particles[i][0] = (self.particles[i][0]).astype(int)
        #    #conv_part[i][0] = int(float(conv_part[i][0]))
        #    print("self particles2: ",conv_part[i][0])
        #    #self.particles[i][1] = int(self.particles[i][1])

        #print("int check: ",self.particles)
        #raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.
        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.particles1 = kwargs.get('particles1',(np.random.rand(self.num_particles,2) * np.shape(self.frame)[0]).astype(int))
        self.template1 = kwargs.get('template1',self.template[0])

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.
        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        #ParticleFilter.process(self,frame)
        #AppearanceModelPF.process(self,frame)


        #        #print("particles appearance: ",self.particles)

        #template = self.template
        ##frame = self.frame

        #temp_y = np.shape(template)[0]
        #temp_x = np.shape(template)[1]

        #particles = self.particles
        #particles_mean = particles.mean(axis=0)

        #weights = self.weights
        ##print("weights: ",weights)

        ##print("template: ",np.shape(template))
        #cv2.imwrite('template.png',template)

        #alpha = 0.1

        #best = frame[int(particles_mean[1]-(temp_y/2)):int(particles_mean[1]+(temp_y/2)),int(particles_mean[0]-(temp_x/2)):int(particles_mean[0]+(temp_x/2)),:]

        #best = cv2.cvtColor(best,cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('best.png',best)
        #template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        
        #if np.shape(best) == np.shape(template):
        #    new_template = alpha*best + (1-alpha)*template
        #else:
        #    new_template = template

        #new_template = new_template.astype('float32')
        #new_template = cv2.cvtColor(new_template,cv2.COLOR_GRAY2BGR)

        ##print("new template: ",np.shape(new_template))
        #cv2.imwrite('new_template.png',new_template)

        #self.template = new_template.astype('float32')

        ##print("particle mean: ",self.particles.mean(axis=0))
        ##print("particle mean2: ",self.particles.mean(axis=1))

        ##for x in frame:
        ##    for y in x:

        ##        frame_cutout = frame[int(y-(temp_y/2)):int(y+(temp_y/2)),int(x-(temp_x/2)):int(x+(temp_x/2)),:]


        ##ParticleFilter.get_error_metric(self, template, frame_cutout)
        ##prev_particles = self.particles
        ##print("prev particles: ",prev_particles)

        ##self.sigma_exp = 0.8*self.sigma_exp
        ##self.sigma_exp = 20
        ##self.num_particles = 5*self.num_particles
        #ParticleFilter.process(self,frame)
        ##print("here")

        ##new_best = self.alpha*self.particles + (1-self.alpha)*prev_particles

        ##self.particles = new_best
        ##conv_part = self.particles
        ##conv_part = conv_part.astype(int)
        ##self.particles = conv_part

        ##print("conv part: ",conv_part)
        ##for i in range(len(conv_part)):
        ##    print("conv part: ",type(conv_part[i]))
        ##    print("self particles1: ",conv_part[i][0])
        ##    conv_part[i] = (conv_part[i]).astype(int)
        ##    #self.particles[i][0] = int(self.particles[i][0])
        ##    #self.particles[i][0] = (self.particles[i][0]).astype(int)
        ##    #conv_part[i][0] = int(float(conv_part[i][0]))
        ##    print("self particles2: ",conv_part[i][0])
        ##    #self.particles[i][1] = int(self.particles[i][1])

        ##print("int check: ",self.particles)




        img1 = np.copy(frame)
        img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        cv2.imwrite("this_temp.png",self.template)

        #print("process frame shape: ",np.shape(img1))

        temp1 = self.template1

        img_temp1 = np.copy(temp1)
        img_temp1 = cv2.cvtColor(img_temp1,cv2.COLOR_BGR2GRAY)

        temp1_size_y = np.shape(temp1)[0]
        temp1_size_x = np.shape(temp1)[1]

        temp1_size_y = np.shape(img_temp1)[0]
        temp1_size_x = np.shape(img_temp1)[1]

        particles1 = self.particles1
        #print("sig dyn: ",self.sigma_dyn)

        weights1 = self.weights1
        #print("particles1: ",particles1)
        #print("old weights1: ",weights1)
        #self.sigma_exp = 5

        mse_vec = []

        for i in range(len(particles1)):
            #print("i: ",i)
            #frame_cutout = img1[int(particles1[i][1]-int(temp1_size_y/2)):int(particles1[i][1]+int(temp1_size_y/2)),int(particles1[i][0]-int(temp1_size_x/2)):int(particles1[i][0]+int(temp1_size_x/2)),:]
            frame_cutout = img1[int(particles1[i][1]-int(temp1_size_y/2)):int(particles1[i][1]+int(temp1_size_y/2)+1),int(particles1[i][0]-int(temp1_size_x/2)):int(particles1[i][0]+int(temp1_size_x/2)+1),:]
            #print("temp1 shape: ",np.shape(temp1))
            #print("fc shape: ",np.shape(frame_cutout))
            #print("frame cutout: ",frame_cutout)
            if np.shape(temp1) == np.shape(frame_cutout):
                #print("here1")
                mse = ParticleFilter.get_error_metric(self,temp1,frame_cutout)
                weights1[i] = np.exp(-mse/(2*self.sigma_exp))
                mse_vec.append(mse)
            else:

                #temp1 = temp1[]
                weights1[i] = 0
                #print("here2")
                #print("cutout shape: ",np.shape(frame_cutout))
                #temp12 = temp1[0:np.shape(frame_cutout)[0],0:np.shape(frame_cutout)[1],:]
                #print("temp12: ",temp12)
                #print("len check: ",np.shape(temp12))
                #for n in range(len(temp12[0])):
                #    print("temp12n: ",temp12[0][n])
                #    temp12[0][n]= [temp12[0][n][0],temp12[0][n][2],temp12[0][n][1]]
                ##print("temp1late2: ",temp12)
                ##print("frame cutout: ",frame_cutout)
                #mse = self.get_error_metric(temp12,frame_cutout)


            #print("mse sum: ",sum(mse_vec))
            #print("max weight: ",max(weights1))

            tot_weights1 = sum(weights1)

            for n in range(len(weights1)):
                weights1[n] = weights1[n] / tot_weights1
            #print("weights1 sum: ",sum(weights1))
            
            #weights1[i] = np.exp(-mse/(2*self.sigma_exp))




        #print("old weights1: ",self.weights1)
        #print("new weights1: ",weights1)

        self.weights1 = weights1

        #print("old particles1: ",self.particles1)
        #print("old particles1 shape: ",np.shape(self.particles1))

        #if sum(mse_vec) < 10000:
        #    new_particles1 = ParticleFilter.resample_particles1(self)
        #else:
        #    new_particles1 = particles1

        new_particles1 = ParticleFilter.resample_particles1(self)
        #new_particles1 = np.stack(new_particles1)


        #print("new particles1: ",new_particles1)
        #print("new particles1 shape: ",np.shape(new_particles1))
        
        self.particles1 = new_particles1

        #if mse 

        #raise NotImplementedError



# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = cv2.imread('test_template_2.jpg')
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))


def run_kalman_filter(kf, imgs_dir, noise, sensor, save_frames={},
                      template_loc=None):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))