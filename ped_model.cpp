//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "cuda_dummy.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <pthread.h>
#include <omp.h>
#define NUM_THREADS 4

struct thread_data{
   int  start;
   int  end;
   std::vector<Ped::Tagent*> *agent;
};

struct lock{
	omp_lock_t border_lock;
	int val1;
	int val2;
	int val3;
};

omp_lock_t border_lock[4];
//for(int j=0; j<4 ;j++) omp_init_lock(&border_lock[j]);

struct thread_data thread_data_array[NUM_THREADS];


struct lock lock_array[NUM_THREADS];


//struct lock lock_array[1]={border_lock[1],399,401,501};
//struct lock lock_array[2]={border_lock[2],499,501,401};
//struct lock lock_array[3]={border_lock[3],399,401,500};

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario)
{
  agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
  treehash = new std::map<const Ped::Tagent*, Ped::Ttree*>();

  // Create a new quadtree containing all agents
  tree = new Ttree(NULL, treehash, 0, treeDepth, 0, 0, 1000, 800);
  for (std::vector<Ped::Tagent*>::iterator it = agents.begin(); it != agents.end(); ++it)
  {
    tree->addAgent(*it);
  }

  // This is the sequential implementation
  implementation = OMPCOL;

  // Set up heatmap (relevant for Assignment 4)
  setupHeatmapSeq();
}

//Pthreads implementation of tick()

extern "C" void* tickP(void *threadarg);

void *tickP(void *threadarg) {
	struct thread_data *my_data;
	my_data = (struct thread_data *) threadarg;
	for (int i=(my_data->start); i<=(my_data->end); i++)
	{
		((my_data->agent)->at(i))->computeNextDesiredPosition();
		((my_data->agent)->at(i))->setX(((my_data->agent)->at(i))->getDesiredX());
		((my_data->agent)->at(i))->setY(((my_data->agent)->at(i))->getDesiredY());
	}
	pthread_exit(NULL);
}

void Ped::Model::tick()

{
	lock_array[0].border_lock=border_lock[0];
	lock_array[0].val1=499;
	lock_array[0].val2=501;
	lock_array[0].val3=400;
	lock_array[1].border_lock=border_lock[1];
	lock_array[1].val1=399;
	lock_array[1].val2=401;
	lock_array[1].val3=501;
	lock_array[2].border_lock=border_lock[0];
	lock_array[2].val1=499;
	lock_array[2].val2=501;
	lock_array[2].val3=401;
	lock_array[3].border_lock=border_lock[0];
	lock_array[3].val1=399;
	lock_array[3].val2=401;
	lock_array[3].val3=499;
	std::vector<Ped::Tagent*>::size_type sz = agents.size();
	int portionsize = sz/ NUM_THREADS;
	std::vector<Ped::Tagent*> region[NUM_THREADS];
	switch (implementation) {
	case OMP:
		// OpenMP
		omp_set_num_threads(NUM_THREADS);
		
		#pragma omp parallel for
		for (int i=0; i<sz; i++)
 		{
			(*agents[i]).computeNextDesiredPosition();
			(*agents[i]).setX((*agents[i]).getDesiredX());
			(*agents[i]).setY((*agents[i]).getDesiredY());
	 	}
		break;

	case PTHREAD:
		// Pthreads
		pthread_t threads[NUM_THREADS];
		pthread_attr_t attr;
		int rc;
		int t;

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		for (t = 0; t<NUM_THREADS; t++) {
			thread_data_array[t].start = t * portionsize;
			thread_data_array[t].end = (t + 1) * portionsize - 1;
			thread_data_array[t].agent = &agents;
			rc = pthread_create(&threads[t], &attr, tickP, (void *)&thread_data_array[t]);
			if (rc) {
				cout << "ERROR; return code from pthread_create() is " << rc << endl;
				exit(-1);
			}
		}
		pthread_attr_destroy(&attr);
		for (t = 0; t<NUM_THREADS; t++) {
			pthread_join(threads[t], NULL);
		}
		break;

	case SEQ:
		// Sequential
		for (std::vector<Ped::Tagent*>::iterator it = agents.begin(); it != agents.end(); ++it)
		{
			(*it)->computeNextDesiredPosition();
			(*it)->setX((*it)->getDesiredX());
			(*it)->setY((*it)->getDesiredY());
		}
		break;
	case SEQCOL:
		//Collision detection sequential
		for (std::vector<Ped::Tagent*>::iterator it = agents.begin(); it != agents.end(); ++it)
		{
			(*it)->computeNextDesiredPosition();
			move(*it);
		}
		break;
	case OMPCOL:
		//Collision detection with OMP
		omp_lock_t region_lock[4];
		for(int j=0; j<4 ;j++) omp_init_lock(&region_lock[j]);
		omp_set_num_threads(NUM_THREADS);
		#pragma omp parallel for 
		for (int i=0; i<sz; i++)
		{
			if ((*agents[i]).getX() <= 500 && (*agents[i]).getY() <= 400){
				omp_set_lock(&region_lock[0]);
				region[0].push_back(agents[i]);
				omp_unset_lock(&region_lock[0]);
			}
			if ((*agents[i]).getX() > 500 && (*agents[i]).getY() <= 400){
				omp_set_lock(&region_lock[1]);
				region[1].push_back(agents[i]);
				omp_unset_lock(&region_lock[1]);
			}
			if ((*agents[i]).getX() > 500 && (*agents[i]).getY() > 400){
				omp_set_lock(&region_lock[2]);
				region[2].push_back(agents[i]);
				omp_unset_lock(&region_lock[2]);
			}
			if ((*agents[i]).getX() <= 500 && (*agents[i]).getY() > 400){
				omp_set_lock(&region_lock[3]);
				region[3].push_back(agents[i]);
				omp_unset_lock(&region_lock[3]);
			}
		}
		for(int j=0; j<4 ;j++) omp_destroy_lock(&region_lock[j]);
		for(int j=0; j<4 ;j++) omp_init_lock(&border_lock[j]);
		#pragma omp parallel
		{
			
			int id,agentX,agentY;
			id=omp_get_thread_num();
			for (std::vector<Ped::Tagent*>::iterator it = region[id].begin(); it != region[id].end(); ++it)
			{
				(*it)->computeNextDesiredPosition();
				agentX=(*it)->getX();
				agentY=(*it)->getY();
					
				if((agentX>=lock_array[0].val1 && agentX<=lock_array[0].val2) && agentY<=lock_array[0].val3){
						omp_set_lock(&lock_array[0].border_lock);
						move(*it);
						omp_unset_lock(&lock_array[0].border_lock);
				}
				if((agentY>=lock_array[1].val1 && agentY<=lock_array[1].val2) && agentX>=lock_array[1].val3){
						omp_set_lock(&lock_array[1].border_lock);
						move(*it);
						omp_unset_lock(&lock_array[1].border_lock);
				}
				if((agentX>=lock_array[2].val1 && agentX<=lock_array[2].val2) && agentY>=lock_array[2].val3){
						omp_set_lock(&lock_array[2].border_lock);
						move(*it);
						omp_unset_lock(&lock_array[2].border_lock);
				}
				if((agentY>=lock_array[3].val1 && agentY<=lock_array[3].val2) && agentX<=lock_array[3].val3){
						omp_set_lock(&lock_array[3].border_lock);
						move(*it);
						omp_unset_lock(&lock_array[3].border_lock);
				}
				
				
					
				else
					move(*it);
			}
		}
		for(int j=0; j<4 ;j++) omp_destroy_lock(&border_lock[j]);
		
		break;

	default:
		break;
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////
void Ped::Model::move( Ped::Tagent *agent)
{
  // Search for neighboring agents
  set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);
    
  // Retrieve their positions
  std::vector<std::pair<int, int> > takenPositions;
  for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
    std::pair<int,int> position((*neighborIt)->getX(), (*neighborIt)->getY());
    takenPositions.push_back(position);
  }

  // Compute the three alternative positions that would bring the agent
  // closer to his desiredPosition, starting with the desiredPosition itself
  std::vector<std::pair<int, int> > prioritizedAlternatives;
  std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
  prioritizedAlternatives.push_back(pDesired);

  int diffX = pDesired.first - agent->getX();
  int diffY = pDesired.second - agent->getY();
  std::pair<int, int> p1, p2;
  if (diffX == 0 || diffY == 0)
  {
    // Agent wants to walk straight to North, South, West or East
    p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
    p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
  }
  else {
    // Agent wants to walk diagonally
    p1 = std::make_pair(pDesired.first, agent->getY());
    p2 = std::make_pair(agent->getX(), pDesired.second);
  }
  prioritizedAlternatives.push_back(p1);
  prioritizedAlternatives.push_back(p2);

  // Find the first empty alternative position
  for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

    // If the current position is not yet taken by any neighbor
    if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

      // Set the agent's position 
      agent->setX((*it).first);
      agent->setY((*it).second);

      // Update the quadtree
      (*treehash)[agent]->moveAgent(agent);
      break;
    }
  }
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
  // if there is no tree, return all agents
  if(tree == NULL) 
    return set<const Ped::Tagent*>(agents.begin(), agents.end());

  // create the output list
  list<const Ped::Tagent*> neighborList;
  getNeighbors(neighborList, x, y, dist);

  // copy the neighbors to a set
  return set<const Ped::Tagent*>(neighborList.begin(), neighborList.end());
}

/// Populates the list of neighbors that can be found around x/y./// This triggers a cleanup of the tree structure. Unused leaf nodes are collected in order to
/// save memory. Ideally cleanup() is called every second, or about every 20 timestep.
/// \date    2012-01-28
void Ped::Model::cleanup() {
  if(tree != NULL)
    tree->cut();
}

/// \date    2012-01-29
/// \param   the list to populate
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
void Ped::Model::getNeighbors(list<const Ped::Tagent*>& neighborList, int x, int y, int dist) const {
  stack<Ped::Ttree*> treestack;

  treestack.push(tree);
  while(!treestack.empty()) {
    Ped::Ttree *t = treestack.top();
    treestack.pop();
    if (t->isleaf) {
      t->getAgents(neighborList);
    }
    else {
      if (t->tree1->intersects(x, y, dist)) treestack.push(t->tree1);
      if (t->tree2->intersects(x, y, dist)) treestack.push(t->tree2);
      if (t->tree3->intersects(x, y, dist)) treestack.push(t->tree3);
      if (t->tree4->intersects(x, y, dist)) treestack.push(t->tree4);
    }
  }
}

Ped::Model::~Model()
{
  if(tree != NULL)
  {
    delete tree;
    tree = NULL;
  }
  if(treehash != NULL)
  {
    delete treehash;
    treehash = NULL;
  }
}
