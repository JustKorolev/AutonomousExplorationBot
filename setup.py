from setuptools import setup
from glob import glob

package_name = 'turtlebot'

files = [('share/' + package_name + '/launch', glob('launch/*')),
         ('share/' + package_name + '/rviz',   glob('rviz/*')),
         ('share/' + package_name + '/maps',   glob('maps/*')),
         ('share/' + package_name + '/urdf',   glob('urdf/*')),
         ('share/' + package_name + '/models', glob('models/*'))]

for bag in glob('bags/*'):
    files.append(('share/' + package_name + '/' + bag, glob(bag + '/*')))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]+files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 133b TurtleBot/Mapping Code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'buildmap          = turtlebot.buildmap:main',
            'slowbuildmap      = turtlebot.slowbuildmap:main',
            'teleop            = turtlebot.teleop:main',
            'auton            = turtlebot.auton:main',
            'noisylocalization = turtlebot.noisylocalization:main',
        ],
    },
)
