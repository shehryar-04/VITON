'use client';

import Image from 'next/image';
import { Globe, Users, ShoppingBag, Sparkles } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-16">
      {/* Hero Section */}
      <div className="text-center mb-16 animate-in fade-in duration-700 slide-in-from-bottom-4">
        <h1 className="text-4xl md:text-5xl font-bold mb-6">About StyleShift</h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Revolutionizing the way you shop for clothes with cutting-edge virtual try-on technology
        </p>
      </div>

      {/* Mission Statement */}
      <section className="grid md:grid-cols-2 gap-12 items-center mb-20 animate-in fade-in duration-700 slide-in-from-bottom-4 delay-200">
        <div className="relative h-[400px] rounded-lg overflow-hidden">
          <Image
            src="/categories/about-img.png"
            alt="Our Mission"
            fill
            className="object-cover"
          />
        </div>
        <div>
          <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
          <p className="text-lg text-muted-foreground mb-6">
            At StyleShift, we're on a mission to transform the online shopping experience. 
            We believe that trying on clothes should be effortless, whether you're in a store 
            or shopping from the comfort of your home.
          </p>
          <p className="text-lg text-muted-foreground">
            Our innovative virtual try-on technology bridges the gap between online and in-store 
            shopping, giving you the confidence to make the right fashion choices every time.
          </p>
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-20 animate-in fade-in duration-700 slide-in-from-bottom-4 delay-300">
        <div className="text-center p-6 rounded-lg bg-card">
          <Globe className="w-12 h-12 mx-auto mb-4 text-primary" />
          <h3 className="text-xl font-semibold mb-2">Global Reach</h3>
          <p className="text-muted-foreground">Serving fashion enthusiasts worldwide with our virtual try-on platform</p>
        </div>
        <div className="text-center p-6 rounded-lg bg-card">
          <Users className="w-12 h-12 mx-auto mb-4 text-primary" />
          <h3 className="text-xl font-semibold mb-2">Happy Customers</h3>
          <p className="text-muted-foreground">Thousands of satisfied customers with perfect fitting clothes</p>
        </div>
        <div className="text-center p-6 rounded-lg bg-card">
          <ShoppingBag className="w-12 h-12 mx-auto mb-4 text-primary" />
          <h3 className="text-xl font-semibold mb-2">Vast Collection</h3>
          <p className="text-muted-foreground">Wide range of styles and brands to choose from</p>
        </div>
        <div className="text-center p-6 rounded-lg bg-card">
          <Sparkles className="w-12 h-12 mx-auto mb-4 text-primary" />
          <h3 className="text-xl font-semibold mb-2">Innovation</h3>
          <p className="text-muted-foreground">Cutting-edge AI technology for accurate virtual try-ons</p>
        </div>
      </section>

      {/* Team Section */}
      <section className="mb-20 animate-in fade-in duration-700 slide-in-from-bottom-4 delay-400">
        <h2 className="text-3xl font-bold text-center mb-12">Meet Our Team</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="relative w-48 h-48 mx-auto mb-4">
              <Image
                src="/categories/usman.jpg"
                alt="Team Member"
                fill
                className="object-cover rounded-full"
              />
            </div>
            <h3 className="text-xl font-semibold mb-2">Muhammad Usman</h3>
            {/* <p className="text-muted-foreground">CEO & Founder</p> */}
          </div>
          <div className="text-center">
            <div className="relative w-48 h-48 mx-auto mb-4">
              <Image
                src="/categories/mudasir1.png"
                alt="Team Member"
                fill
                className="object-cover rounded-full"
              />
            </div>
            <h3 className="text-xl font-semibold mb-2">Mudassir Azhar</h3>
            {/* <p className="text-muted-foreground">CTO</p> */}
          </div>
          <div className="text-center">
            <div className="relative w-48 h-48 mx-auto mb-4">
              <Image
                src="/categories/alishba.jpg"
                alt="Team Member"
                fill
                className="object-cover rounded-full"
              />
            </div>
            <h3 className="text-xl font-semibold mb-2">Alishba Fatima</h3>
            {/* <p className="text-muted-foreground">Head of Design</p> */}
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="bg-card rounded-lg p-12 text-center animate-in fade-in duration-700 slide-in-from-bottom-4 delay-500">
        <h2 className="text-3xl font-bold mb-8">Our Values</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-4">Innovation</h3>
            <p className="text-muted-foreground">
              We constantly push the boundaries of technology to improve your shopping experience
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Customer First</h3>
            <p className="text-muted-foreground">
              Your satisfaction and confidence in your purchases are our top priorities
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Sustainability</h3>
            <p className="text-muted-foreground">
              We're committed to reducing returns and promoting sustainable fashion choices
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
