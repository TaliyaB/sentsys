from django.urls import path, include
from django.contrib import admin
from . import views

urlpatterns = [
    path('', views.upload, name='up'),
    path('up', views.upload, name='up'),
    path('visuals', views.visuals, name='visuals'),
    path('data', views.data, name='data'),
    path('lda', views.lda, name='lda'),
    path('nouns0', views.nouns0, name='nouns0'),
    path('nouns1', views.nouns1, name='nouns1'),
    path('nouns2', views.nouns2, name='nouns2'),
    path('adj0', views.adj0, name='adj0'),
    path('adj1', views.adj1, name='adj1'),
    path('adj2', views.adj2, name='adj2'),
    path('poslda', views.poslda, name='poslda'),
    path('neglda', views.neglda, name='neglda'),
    path('admin/', admin.site.urls),
]
