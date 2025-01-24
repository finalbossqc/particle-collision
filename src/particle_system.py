import numpy as np
import pygame
import sys

class ParticleSystem:
	def __init__(self, window_size, N=20,damp=0.9, anim=1):
		self.N = N
		self.damp = damp
		self.dt = 0
		self.masses = []
		self.positions = []
		self.velocities = []
		self.radii = []
		self.window_size = window_size
		self.g = np.array([0,-9.8])
		self.anim = anim
		self.frames = 0

	@property
	def size(self):
		return len(self.masses)	

	def mass(self, i):
		return self.masses[i]

	def radius(self, i):
		return self.radii[i]

	def position(self, i):
		return self.positions[i]

	def velocity(self, i):
		return self.velocities[i]

	def add_particle(self, mass, position, velocity, r):
		self.masses.append(mass)
		self.positions.append(np.array(position))
		self.velocities.append(np.array(velocity))
		self.radii.append(r)

	def check_collision(self, i, j):
		p1,r1 = self.position(i), self.radius(i)
		p2,r2 = self.position(j), self.radius(j)
		d=np.linalg.norm(p2 - p1)
		return d < r1 + r2

	def compute_velocities(self, i, j, dt):
		p1,v1,m1,r1 = self.position(i), self.velocity(i), self.mass(i), self.radius(i)
		p2,v2,m2,r2 = self.position(j), self.velocity(j), self.mass(j), self.radius(j)

		M = m1 + m2
		n = np.linalg.norm(p1 - p2) ** 2

		u1 = ( v1 - 2 * m2 / M * np.dot(v1 - v2, p1 - p2) / n * (p1 - p2) ) * self.damp
		u2 = ( v2 - 2 * m1 / M * np.dot(v2 - v1, p2 - p1) / n * (p2 - p1) ) * self.damp

		self.velocities[i] = u1
		self.velocities[j] = u2

	def compute_positions(self, i, j):
		p1,v1,m1,r1 = self.position(i), self.velocity(i), self.mass(i), self.radius(i)
		p2,v2,m2,r2 = self.position(j), self.velocity(j), self.mass(j), self.radius(j)

		p1 = p1
		p2 = p2

		f1 = np.linalg.norm(v1)
		f2 = np.linalg.norm(v2)
		F = f1 + f2

		d = np.linalg.norm((p2 + v2 * self.dt) - (p1 + v1 * self.dt))
		e = p2 - p1

		a1 = np.arctan2(e[1], e[0])
		a2 = a1 + np.pi

		if d < r1 + r2:
			n1 = p1 + np.array((np.cos(a2), np.sin(a2))) * (r1 + r2 - d) * 2 * f1 / F
			n2 = p2 + np.array((np.cos(a1), np.sin(a1))) * (r1 + r2 - d) * 2 * f2 / F
			self.positions[i] = n1
			self.positions[j] = n2

	def update_position(self, i):
		w,h = self.window_size
		
		p,v,m,r = self.position(i), self.velocity(i), self.mass(i), self.radius(i)
		p = p + v * self.dt - 0.5*self.g*self.dt**2
		v = v - self.g*self.dt

		x,y = p
		dx,dy = v

		if x - r < 0:
			x = r
			dx = -self.damp*dx

		if x + r > w:
			x = w - r
			dx = -self.damp*dx

		if y - r < 0:
			y = r
			dy = -self.damp*dy

		if y + r > h:
			y = h - r
			dy = -self.damp*dy

		v = np.array((dx,dy))
		p = np.array((x,y))

		self.velocities[i] = v
		self.positions[i] = p

	def compute(self, dt):
		self.dt = dt
		
		for i in range(self.size):
			for j in range(i, self.size):
				if i != j and self.check_collision(i,j):
					self.compute_velocities(i,j,dt)
					self.compute_positions(i,j)
		
		for i in range(self.size):
			self.update_position(i)

	def draw(self, screen):
		for i in range(self.size):
			p,r = self.position(i), self.radius(i)
			pygame.draw.circle(screen, (255, 255, 255), [int(x) for x in p], int(r))
		pygame.image.save(screen, f"frames/frame_{self.frames:04d}.png")
		self.frames += 1
			
	def run(self):
		if (int(anim) == 1):
			screenw,screenh = self.window_size

			screen = pygame.display.set_mode((screenw, screenh))
			clock = pygame.time.Clock()

			system=ParticleSystem((screenw, screenh))

			for i in range(int(self.N)):
				m = np.random.randint(50, 100)
				r = np.random.uniform(10, 50)
				
				p = [np.random.uniform(screenw), 
					np.random.uniform(screenh)]
				
				v = [np.random.uniform(low=-600, high=600), 
					np.random.uniform(low=-600, high=600)]

				system.add_particle(m,p,v,r)	

			running = True
			time = 0

			with open("sim.out", "w") as f:
				print("Particle data", file=f)
			
			while running:
				with open("sim.out", "a") as f:
					print("time_step="+str(time), file=f)
					print("x="+str(system.positions), file=f)
					print("r="+str(system.radii), file=f)
				
				dt = clock.tick() / 1000
				time += dt

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						running = False

				screen.fill((0,0,0))
				
				system.compute(dt)
				system.draw(screen)
				
				pygame.display.flip()
			
			pygame.quit()
		else:
			screenw,screenh = self.window_size

			screen = pygame.display.set_mode((screenw, screenh))
			clock = pygame.time.Clock()

			system=ParticleSystem((screenw, screenh))

			for i in range(int(self.N)):

				m = np.random.randint(50, 100)
				r = np.random.uniform(10, 50)
				
				p = [np.random.uniform(screenw), 
					np.random.uniform(screenh)]
				
				v = [np.random.uniform(low=-600, high=600), 
					np.random.uniform(low=-600, high=600)]

				system.add_particle(m,p,v,r)	

			running = True
			time = 0

			with open("sim.out", "w") as f:
				print("Particle data", file=f)
			
			while running:
				with open("sim.out", "a") as f:
					print("time_step="+str(time), file=f)
					print("x="+str(system.positions), file=f)
					print("r="+str(system.radii), file=f)

				dt = clock.tick() / 1000
				time += dt
				system.compute(dt)

if __name__=="__main__":
	N = 5
	damp = 1
	anim = 1
	for i in range(len(sys.argv)):
		if (sys.argv[i] == "-N"):
			try:
				N = sys.argv[i+1]
			except:
				pass
				
		if (sys.argv[i] == "-d"):
			try:
				damp = sys.argv[i+1]
			except:
				pass
		
		if (sys.argv[i] == "-a"):
			try:
				anim = sys.argv[i+1]
			except:
				pass
		
	print(anim)

	ParticleSystem(window_size=(1200, 800), N=N, damp=damp, anim=anim).run()
